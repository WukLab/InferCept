import enum
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager, Device
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from vllm.core.solver import Solver
from heapq import *
from functools import total_ordering

logger = init_logger(__name__)
QS_THRSH = 16

class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    3. Preservation: keep blocks of the PAUSED sequences ONLY
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()
    PRESERVE = enum.auto()
    PARTIAL = enum.auto()           # if needed we can use this instead

class DiscardMode(enum.Enum):
    SEQUENCE = enum.auto()
    FIRST_TOKEN = enum.auto()
    LAST_TOKEN = enum.auto()
    
class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
        is_prompt: List[bool] = None,
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        self.ignored_seq_groups = ignored_seq_groups
        self.is_prompt = is_prompt if is_prompt else [prompt_run] * len(scheduled_seq_groups)
        assert len(self.is_prompt) == len(self.scheduled_seq_groups)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

@total_ordering
class ResumeChunk:
    def __init__(self, request_id, c_d, c_s, chunk_idx, start_est) -> None:
        self.request_id = request_id
        self.c_d = c_d
        self.c_s = c_s
        self.chunk_idx = chunk_idx
        self.start_est = start_est

    def __eq__(self, other):
        return self.request_id == other.request_id and self.start_est == other.start_est and self.chunk_idx == other.chunk_idx

    def __lt__(self, other):
        if self.start_est == other.start_est:
            return self.chunk_idx < other.chunk_idx
        return self.start_est < other.start_est

class ResumeQueue:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.free_recompute_tokens = 128                       # FIXME get this from profile
        self.free_swap_tokens = 976                            # FIXME get this from profile
        self._queue: List[List[ResumeChunk]] = []
        self._future: List[ResumeChunk] = []
        self.correction_offset: float = 0

    def add(self, iter: int, chunk: ResumeChunk) -> None:
        if iter < len(self._queue):
            self._queue[iter].append(chunk)
        else:
            self._queue.append([chunk])
    
    def add_future(self, chunk: ResumeChunk) -> None:
        heappush(self._future, chunk)

    def activate_chunks(self, running: int, request_id: int, now: float, iter_time: float) -> None:
        # removes from future queue and moves into current
        start, iter = self._get_start_time(now, running)
        tail_base_time = self.get_tail_time(now)

        curr = 0
        while curr < len(self._future):
            chunk = self._future[curr]
            if chunk.request_id == request_id:
                if iter < len(self._queue):
                    chunk.start_est = start
                else:
                    chunk.start_est = tail_base_time + iter_time
                    tail_base_time = chunk.start_est
                self.add(iter, chunk)
                del self._future[curr]
            else:
                curr += 1

    def get_iter_chunks(self, iter: int) -> Tuple[int, int]:
        if iter < len(self._queue) and iter >= 0:
            return self._get_iter_free_chunks(self._queue[iter])
        return (0, 0)
    
    def _get_iter_free_chunks(self, chunks: List[ResumeChunk]) -> Tuple[int, int]:
        acc_d = 0
        acc_s = 0
        for chunk in chunks:
            acc_d += chunk.c_d
            acc_s += chunk.c_s
        return (acc_d, acc_s)

    def get_tail_time(self, now):
        if self._queue:
            return self._queue[-1][0].start_est + self.correction_offset
        return now
    
    def _get_start_time(self, now: float, running: int):
        for i, chunks in enumerate(self._queue):
            free_d, free_s = self._get_iter_free_chunks(chunks)
            if i == len(self._queue)-1 or running + free_d * self.block_size < self.free_recompute_tokens and free_s * self.block_size < self.free_swap_tokens:
                return (chunks[0].start_est + self.correction_offset, i)
        return now, len(self._queue)

    def get_start_time(self, now: float, running: int, api_exec_time: float) -> Tuple[float, int]:
        if self.get_tail_time(now) < now + api_exec_time:
            return self._get_start_time(now, running)
        else:
            return now + api_exec_time, None
        
class IterStat:
    def __init__(
        self,
        paused_gpu_blocks,
        not_involved_gpu_blocks,
        batch_size,
        recomp_token_size,
        recomp_mem_blocks,
        swap_in_blocks,
        swap_out_blocks,
    ):
        self.paused_gpu_blocks = paused_gpu_blocks
        self.not_involved_gpu_blocks = not_involved_gpu_blocks
        self.batch_size = batch_size
        self.recomp_token_size = recomp_token_size
        self.recomp_mem_blocks = recomp_mem_blocks
        self.swap_in_blocks = swap_in_blocks
        self.swap_out_blocks = swap_out_blocks
        self.iter_time = 0.0

    def __repr__(self) -> str:
        return f'{self.paused_gpu_blocks}, {self.not_involved_gpu_blocks}, {self.batch_size}, {self.recomp_token_size}, {self.recomp_mem_blocks}, {self.swap_in_blocks}, {self.swap_out_blocks}, {self.iter_time}'

class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.total_tks = 0

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
        # Sequence groups in the PAUSED state.
        # seq_group_id -> (group, preemption_mode)
        self.paused: Dict[str, Tuple(SequenceGroup, PreemptionMode)] = {}
        # Sequence groups that are resuming
        self.resuming : List[SequenceGroup] = []
        # Sequence groups that have everything in the GPU but not scheduled in the last iter
        self.delayed_running: List[SequenceGroup] = []
        self.total_batched_tokens = 0
        self.history: List[Tuple[float, float, float, int, int, int, PreemptionMode]] = []
        # initialize solver and resume queue

        # TODO convert to profile object/config of some sort
        self.free_recompute_tokens = 128                       # FIXME get this from profile
        self.free_swap_tokens = 640                            # FIXME get this from profile
        self.per_token_swap_latency = 4E-05                    # FIXME get this from profile
        self.batch_polynomial = (1.3E-05, 0.328, 24.1)         # FIXME get this from profile
        self.solver = Solver(block_size=self.cache_config.block_size,
                             free_swap_tokens=self.free_swap_tokens,
                             per_token_swap_latency=self.per_token_swap_latency,
                             batch_polynomial=self.batch_polynomial)
        self.resume_queue = ResumeQueue(block_size=self.block_manager.block_size)
        self.end_time: float = -1              # fix the time at the end of every iteration
        self.free_swap_out_left = 0

        # virtual scheduler
        self.virtual_swap_idx = 0
        self.virtual_waiting_idx = 0
        self.virtual_running_idx = 0
        self.virtual_free_blocks = self.block_manager.get_num_free_gpu_blocks()
        self.debug_iter = 0
        
        self.batch_max_tokens = 0
        self.max_swap_out_blocks = 0
        self.max_swap_in_blocks = 0
        self.paused_gpu_blocks = []
        self.iter_history: List[IterStat] = []

        
        assert self.scheduler_config.api_policy != 'V', 'V policy should use new scheduler'
            
        if self.scheduler_config.api_policy == 'G':
            assert self.scheduler_config.chunk_fill, 'G policy only works with chunk fill to enforce limit'

        # heuristic
        self.already_swapped_paused = False
        

    def f(self, query_chunks, swap_chunks):
        query_tokens = len(self.running) + query_chunks * self.block_manager.block_size
        swap_tokens = swap_chunks * self.block_manager.block_size
        recomp = (self.batch_polynomial[0]*query_tokens**2 + self.batch_polynomial[1]*query_tokens + self.batch_polynomial[2]) / 1000 # ms -> sec
        swap = self.per_token_swap_latency * max(swap_tokens - self.free_swap_tokens, 0)   # sec
        return recomp + swap

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)
    
    def passive_discard(self, required_blocks: int) -> int:
        if not self.paused:
            return 0
        discarded = 0
        while discarded < required_blocks:
            preserved = [group for group, mode in self.paused.values() if mode is PreemptionMode.PRESERVE]
            if not preserved:
                break
            #NOTE: for intra-req discard, this will find the longest api seq
            target: SequenceGroup = min(preserved)
            assert target.num_seqs() == 1
            seq = target.get_seqs()[0]
            discard_num_toks = self.block_manager.free_fst_blk_tokens(seq, 4)
            seq.discard_idx += discard_num_toks
            if self.block_manager.block_tables.get(seq.seq_id, None) is None:
                seq.discard_idx = 0
                seq.data.logical_query_len = seq.data.get_len()
                self.paused[target.request_id]= (target, PreemptionMode.RECOMPUTE)
        return discarded
    
    def virtual_passive_discard(self, required_blocks: int, api_time_threshold: float = 12.0) -> bool:
        discarded = 0
        long_preserved = [group for group, mode in self.paused.values() if mode is PreemptionMode.PRESERVE and group.sampling_params.api_exec_time >= api_time_threshold]
        if not long_preserved:
            return False
        for target in long_preserved:
            assert target.num_seqs() == 1
            seq = target.get_seqs()[0]
            num_blocks_can_discard = len(self.block_manager.get_block_table(seq))
            discarded += num_blocks_can_discard
            if discarded >= required_blocks:
                return True
        return False

    def get_tokens_have_seen(self) -> int:
        groups = self.running + self.delayed_running + self.resuming + self.swapped + self.waiting + [group for group, _ in self.paused.values()]
        return sum(seq.data.seen_tokens for seq_group in groups for seq in seq_group.get_seqs())
    
    # NOTE: SWAP is added to resuming group to reuse chunked swap-in logic
    # TODO: cleanup swap logic
    def pause_seq_group(self, seq_group: SequenceGroup, blocks_to_swapout: Dict[int, int], first_n_blocks = -1) -> None:
        mode = None
        seq_group.sampling_params.api_call_time = time.monotonic()
        # if seq_group.sampling_params.api_exec_time_unknown
        if self.scheduler_config.api_policy == 'H':
            if seq_group.sampling_params.api_exec_time >= self.scheduler_config.heuristic_coef:
                mode = PreemptionMode.SWAP
            else:
                mode = PreemptionMode.PRESERVE
        elif self.scheduler_config.api_policy == 'D':
            mode = PreemptionMode.RECOMPUTE
        elif self.scheduler_config.api_policy == 'S':
            mode = PreemptionMode.SWAP
        elif self.scheduler_config.api_policy == 'P':
            mode = PreemptionMode.PRESERVE
        else:
            raise ValueError(f"Invalid API policy {self.scheduler_config.api_policy}")
        # Deal with it
        if mode is PreemptionMode.PRESERVE:
            self.paused[seq_group.request_id] = (seq_group, PreemptionMode.PRESERVE)
        if mode is PreemptionMode.RECOMPUTE:
            for seq in seq_group.get_seqs():
                seq.data.clear_workload()
                seq.data.discard_start_idx = seq.data.discard_length = 0
                seq.data.swap_start_idx = seq.data.swap_length = 0
                seq.data.inflight_length = seq.data.get_len()
                self.block_manager.free(seq) 
            seq_group.discarded = True
            self.paused[seq_group.request_id] = (seq_group, PreemptionMode.RECOMPUTE)
        if mode is PreemptionMode.PARTIAL:
            self.paused[seq_group.request_id] = (seq_group, PreemptionMode.PARTIAL)
        if mode is PreemptionMode.SWAP:
            if self.block_manager.can_swap_out(seq_group, first_n_blocks):
                gpu_to_cpu, all_swapped = self.block_manager.swap_out(seq_group, status=SequenceStatus.PAUSED_API, first_n_blocks=first_n_blocks)
                blocks_to_swapout.update(gpu_to_cpu)
                seq = seq_group.get_seqs()[0]
                seq.data.clear_workload()
                seq.data.discard_start_idx = seq.data.discard_length = 0
                seq.data.swap_start_idx = 0
                seq.data.swap_length = len(self.block_manager.get_block_table(seq))
                self.paused[seq_group.request_id] = (seq_group, PreemptionMode.SWAP)
            else:
                for seq in seq_group.get_seqs():
                    seq.data.clear_workload()
                    seq.data.discard_start_idx = seq.data.discard_length = 0
                    seq.data.swap_start_idx = seq.data.swap_length = 0
                    seq.data.inflight_length = seq.data.get_len()
                    self.block_manager.free(seq) 
                self.paused[seq_group.request_id] = (seq_group, PreemptionMode.RECOMPUTE)
        return

    def get_paused_seq_group(self, request_id: str) \
        -> Tuple[SequenceGroup, PreemptionMode]:
        if request_id in self.paused:
            return self.paused[request_id]
        else:
            raise ValueError(f"No paused request with request id {request_id}")

    # TODO: (Need discussion) different policy of resuming APIs:
    #  1. (implememted) API with preserve mode: FCFS in the running group
    #  2. (?) API with recompute mode: First in the waiting group
    #  3. (?) API with swap mode: FCFS in the swapped group
    def resume_seq_group(self, request_id: str) -> None:
        seq_group, mode = self.get_paused_seq_group(request_id)
        # If not all seqs are resumed, do nothing
        if seq_group.is_paused():
            return
        del self.paused[request_id]
        if mode is PreemptionMode.PRESERVE:
            for seq in seq_group.get_seqs(status=SequenceStatus.RESUMED_API):
                seq.status = SequenceStatus.RUNNING
            self.running.append(seq_group)
        elif mode is PreemptionMode.SWAP:
            for seq in seq_group.get_seqs(status=SequenceStatus.RESUMED_API):
                seq.status = SequenceStatus.SWAPPED
            self.swapped.append(seq_group)
        elif mode is PreemptionMode.RECOMPUTE:
            for seq in seq_group.get_seqs(status=SequenceStatus.RESUMED_API):
                seq.status = SequenceStatus.WAITING
            self.waiting.append(seq_group) # switch it for vanilla discard
        elif mode is PreemptionMode.PARTIAL:
            for seq in seq_group.get_seqs(status=SequenceStatus.RESUMED_API):
                seq.status = SequenceStatus.RESUMING
            self.resuming.append(seq_group)
        else:
            raise NotImplementedError("Pause only support preserving now")

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.paused or self.resuming

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + len(self.paused)


    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        # logger.info(f'iter {self.debug_iter}')

        # reset virtual scheduler
        self.virtual_swap_idx = 0
        self.virtual_waiting_idx = 0
        self.virtual_running_idx = 0
        self.virtual_free_blocks = self.block_manager.get_num_free_gpu_blocks()
        self.already_swapped_paused = False

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        # if not self.swapped and not self.resuming:
        if not self.resuming and not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            num_batched_tokens = 0
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            # self.waiting = self.policy.sort_by_priority(now, self.waiting)
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt {seq_group.request_id}: ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue
                
                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break
                seq = seq_group.get_seqs()[0]
                seq.data.populate_workload(0, 0, seq.data.inflight_length)
                if not self.block_manager.can_fulfill_blocks(self.block_manager.get_memory_requirement(seq_group)):
                    seq.data.clear_workload()
                    break
                # If the sequence group cannot be allocated, stop.
                #TODO: why swap here
                # ## add check for swap
                # sum_swapped = 0
                # for sg in self.swapped:
                #     num_sg_blocks = len(self.block_manager.get_block_table(sg.get_seqs()[0]))
                #     sum_swapped += num_sg_blocks

                # num_required_blocks = len(seq_group.get_seqs()[0].logical_token_blocks) 
                # num_true_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks() - self.block_manager.watermark_blocks
                # if (num_true_free_gpu_blocks - sum_swapped < num_required_blocks):
                #     break

                seq_group = self.waiting.pop(0)
                seq = seq_group.get_seqs()[0]
                self.block_manager.allocate(seq_group, blocks_to_swap_in)
                assert not len(blocks_to_swap_in), "swap in should be empty"
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)
            
            self.total_tks += num_batched_tokens

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                self.total_batched_tokens += num_batched_tokens
                    
                self.virtual_free_blocks = self.block_manager.get_num_free_gpu_blocks()
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        delayed_running: List[SequenceGroup] = []
        num_batched_tokens = 0
        while self.running:
            seq_group = self.running.pop(0)
            if num_batched_tokens + seq_group.get_seqs()[0].data.inflight_length > self.scheduler_config.max_num_batched_tokens:
                delayed_running.append(seq_group)
                continue
            seq = seq_group.get_seqs()[0]
            seq.data.populate_workload(0, 0, seq.data.inflight_length)
            new_blocks = self.block_manager.get_memory_requirement(seq_group)
            while not self.block_manager.can_fulfill_blocks(new_blocks):
                if delayed_running:
                    victim_seq_group = delayed_running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                num_batched_tokens += seq.data.running_inflight_tokens
                self.block_manager.allocate(seq_group, blocks_to_swap_in)
                seq.status = SequenceStatus.RUNNING
                running.append(seq_group)
        self.running = running
        self.delayed_running = delayed_running
        
        # TODO: resuming group preemption is problematic
        # Contain resumed seqs that are partially discarded or paritially swapped or both
        # Once they are entirely resumed, they will be moved to running group
        self.resuming = self.policy.sort_by_priority(now, self.resuming)
        not_running = []
        while self.resuming:
            seq = seq_group.get_seqs()[0]
            seq_group = self.resuming.pop(0)
            # If the sequence group cannot be swapped in and allocated, evict other resumed
            seq.data.populate_workload(seq.data.discard_length, seq.data.swap_length, seq.data.inflight_length)
            required_blocks = self.block_manager.get_memory_requirement(seq_group)
            while not (self.block_manager.can_fulfill_blocks(required_blocks)):
                if self.resuming:
                    victim_seq_group = self.resuming.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                self.block_manager.allocate(seq_group, blocks_to_swap_in)
                seq = seq_group.get_seqs()[0]
                if seq.data.resume_discard_tokens + seq.data.running_inflight_tokens > 0:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq_group) # if no running_query, only swap in
                else:
                    not_running.append(seq_group)
        self.resuming = not_running
        
        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_batched_tokens = sum(seq_group.get_seqs()[0].data.running_inflight_tokens for seq_group in self.running)
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            while self.swapped:
                seq_group = self.swapped[0]
                seq = seq_group.get_seqs()[0]
                seq.data.populate_workload(0, seq.data.swap_length, seq.data.inflight_length)
                required_mem = self.block_manager.get_memory_requirement(seq_group)
                # If the sequence group cannot be swapped in, stop.
                if not (self.block_manager.can_fulfill_blocks(required_mem)):
                    break
                else:
                    for seq in seq_group.get_seqs(): #TODO: real kernel
                        seq.discard_idx = 0
                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break
                if (num_batched_tokens + seq_group.get_seqs()[0].data.inflight_length > self.scheduler_config.max_num_batched_tokens):
                    # logger.info(f"batched tokens: {num_batched_tokens}, max: {self.scheduler_config.max_num_batched_tokens}")
                    break
                seq_group = self.swapped.pop(0)
                self.block_manager.allocate(seq_group, blocks_to_swap_in)
                for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                    seq.status = SequenceStatus.RUNNING
                    num_batched_tokens += seq.data.running_inflight_tokens
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq.data.running_inflight_tokens
            for seq_group in self.running
                for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING))
        self.total_tks += num_batched_tokens
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        self.virtual_free_blocks = self.block_manager.get_num_free_gpu_blocks()
        return scheduler_outputs
    
    # TODO: fill in the actual implementation
    # input: scheduled normal decoding seq groups
    # return: (max multi-token computation, max swap blocks)
    def _get_max_ragged_batch(self, 
                              running_batch_config: List[SequenceGroup], 
                              swap_in: Dict[int, int], 
                              swap_out: Dict[int, int]) -> Tuple[int, int]:
        num_batched_tokens = sum(seq_group.get_seqs()[0].data.running_query_len for seq_group in running_batch_config)
        max_multi_token_compute = max(0, self.scheduler_config.max_num_batched_tokens)
        max_swap_limit = max(0, 999//self.block_manager.block_size - len(swap_in) - len(swap_out))
        return 512 - num_batched_tokens, max_swap_limit
        # return self.scheduler_config.max_num_batched_tokens, 100
    
    def _if_repeat(self, group_list: List[SequenceGroup]) -> bool:
        repeat = set()
        for group in group_list:
            if group.request_id in repeat:
                return True
            repeat.add(group.request_id)
        return False
    
    # Schedule running group first and fill in chunks
    #TODO: discard from front
    def _schedule_chunk_and_fill(self) -> SchedulerOutputs:
        # logger.info(f'debug iter: {self.debug_iter}')
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        
        # Fix the current time.
        now = time.monotonic()

        if self.scheduler_config.api_policy == 'G':
            saturating_point_toks, free_swap_cap = self._get_max_ragged_batch([], [], [])
            saturating_point_toks = 0
            # for swapping out only
            free_swap_cap = free_swap_cap // 2
            preemption_targets = [seq_group for seq_group, mode in self.paused.values() if (mode is PreemptionMode.PRESERVE and seq_group.api_remaining_time(now) >= 94e-3) or mode is PreemptionMode.PARTIAL]
            # preemption_targets = [seq_group for seq_group, mode in self.paused.values() if (mode is PreemptionMode.PRESERVE or mode is PreemptionMode.PARTIAL)]
            preemption_targets = PolicyFactory.get_policy('lra').sort_by_priority(now, preemption_targets)
            for seq_group in preemption_targets:
                if free_swap_cap + saturating_point_toks <= 0:
                    break
                seq = seq_group.get_seqs()[0]
                gpu_blocks_for_preemption = len(self.block_manager.get_seq_blocks_by_device(seq_group, Device.GPU))
                # plan in number of blocks
                n_discard_blocks, n_swap_out = 0, 0
                if seq.swap_length > 0:
                    # has swap blocks, can only do further swap
                    n_swap_out = min(free_swap_cap, gpu_blocks_for_preemption)
                else:
                    # first exhause discarding limit, since this creates more chance for further discard
                    n_discard_blocks = min(self.block_manager.token_2_block(saturating_point_toks), gpu_blocks_for_preemption)
                    n_swap_out = min(free_swap_cap, gpu_blocks_for_preemption - n_discard_blocks)
                assert n_discard_blocks == 0
                if n_discard_blocks > 0:
                    assert seq.swap_length == 0, 'cannot discard when already swapped'
                    discarded = self.block_manager.free_fst_blk_tokens(seq, n_discard_blocks)
                    assert discarded == n_discard_blocks
                    # IF discard all physical, turn into a waiting seq
                    if seq.seq_id not in self.block_manager.block_tables:
                        saturating_point_toks -= seq.data.get_len() - seq.discard_length
                        seq.data.logical_query_len = seq.data.get_len()
                        seq.discard_start_idx = 0
                        seq.discard_length = 0
                        self.paused[seq_group.request_id] = (seq_group, PreemptionMode.RECOMPUTE)
                        continue
                    else:
                        # Not discarding all, which means discarded blocks are full
                        seq.discard_length += discarded * self.block_manager.block_size
                        saturating_point_toks -= discarded * self.block_manager.block_size
                if n_swap_out > 0:
                    assert len(self.block_manager.get_seq_blocks_by_device(seq_group, Device.GPU)) >= n_swap_out
                    assert self.block_manager.can_swap_out(seq_group, n_swap_out)
                    gpu_to_cpu, _ = self.block_manager.swap_out(seq_group, status=SequenceStatus.PAUSED_API, first_n_blocks=n_swap_out)
                    assert len(gpu_to_cpu) == n_swap_out
                    seq.swap_start_idx = n_discard_blocks
                    seq.swap_length += len(gpu_to_cpu)
                    free_swap_cap -= n_swap_out
                    blocks_to_swap_out.update(gpu_to_cpu)
                    
                self.paused[seq_group.request_id] = (seq_group, PreemptionMode.PARTIAL)
            
        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        is_prompt: List[bool] = []
        self.running = self.policy.sort_by_priority(now, self.running)
        delayed_running: List[SequenceGroup] = []
        
        num_batched_tokens = 0
        while self.running:
            seq_group = self.running.pop(0)
            # tokens that will be forwarded if scheduled
            inflight_tokens = seq_group.get_seqs()[0].data.logical_query_len
            max_mul, _ = self._get_max_ragged_batch(running, blocks_to_swap_in, blocks_to_swap_out)
            inflight_tokens = min(inflight_tokens, max_mul)
            if inflight_tokens <= 0:
                delayed_running.append(seq_group)
                delayed_running.extend(self.running)
                break
            if inflight_tokens + num_batched_tokens > self.scheduler_config.max_num_batched_tokens:
                delayed_running.append(seq_group)
                continue
            new_blocks = self.block_manager.required_new_blocks(seq_group, inflight_tokens)
            # passive discard for running
            # if self.scheduler_config.api_policy == 'V':
            #     if not self.block_manager.can_fulfill_blocks(new_blocks):
            #         free_gpu = self.block_manager.get_num_free_gpu_blocks()
            #         if self.virtual_passive_discard(free_gpu - new_blocks, 0.0):
            #             self.passive_discard(free_gpu - new_blocks)
            while not self.block_manager.can_fulfill_blocks(new_blocks):
                if delayed_running:
                    victim_seq_group = delayed_running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                    seq.data.running_query_len = inflight_tokens
                    seq.data.running_start_idx = seq.data.get_len() - seq.data.logical_query_len
                    num_batched_tokens += seq.data.running_query_len
                # this will allocate by chunks, set running_query_len before calling this
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
                is_prompt.append(False)
        self.running = running
        self.delayed_running = delayed_running
        self.resuming = self.policy.sort_by_priority(now, self.resuming)
        # logger.info(f'iter {self.debug_iter}, resuming: {[seq_group.request_id for seq_group in self.resuming]}')
        not_running = []
        num_batched_tokens = sum(seq_group.get_seqs()[0].data.running_query_len for seq_group in self.running)
        delayed_resuming = []
        while self.resuming:
            seq_group = self.resuming.pop(0)
            # If the sequence group cannot be swapped in and allocated, evict other resumed
            max_mul, max_swap = self._get_max_ragged_batch(running, blocks_to_swap_in, blocks_to_swap_out)
            inflight_tokens = min(seq_group.get_seqs()[0].discard_length, max_mul)
            swap_blocks = min(seq_group.get_seqs()[0].swap_length, max_swap)
            if inflight_tokens + num_batched_tokens > self.scheduler_config.max_num_batched_tokens:
                delayed_resuming.append(seq_group)
                continue
            required_blocks = self.block_manager.required_resume_blocks(seq_group, swap_blocks, inflight_tokens)
            
            while not (self.block_manager.can_fulfill_blocks(required_blocks)):
                # TODO: consider preempt not running as well
                if delayed_resuming:
                    victim_seq_group = delayed_resuming.pop(-1)
                    self._preempt_resuming(victim_seq_group, blocks_to_swap_out, not_running, PreemptionMode.RECOMPUTE)
                    preempted.append(victim_seq_group)
                if self.resuming:
                    victim_seq_group = self.resuming.pop(-1)
                    self._preempt_resuming(victim_seq_group, blocks_to_swap_out, not_running, PreemptionMode.RECOMPUTE)
                    preempted.append(victim_seq_group)
                else:
                    self._preempt_resuming(seq_group, blocks_to_swap_out, not_running, PreemptionMode.RECOMPUTE)
                    preempted.append(seq_group)
                    break
            else:
                resume_swapped = self._swap_in(seq_group, blocks_to_swap_in, swap_blocks, status=SequenceStatus.RESUMING)
                if resume_swapped != swap_blocks:
                    logger.info(f'iter {self.debug_iter}, swap in {swap_blocks}, but resumsed {resume_swapped}')
                assert resume_swapped == swap_blocks, f'should swap {swap_blocks}, but resumsed {resume_swapped}'
                seq = seq_group.get_seqs()[0]
                seq.swap_start_idx += resume_swapped
                seq.swap_length -= resume_swapped
                seq.data.running_query_len = inflight_tokens
                # remember to set running_query_len before calling this
                if seq.data.running_query_len > 0:
                    for seq in seq_group.get_seqs(status=SequenceStatus.RESUMING):
                        seq.status = SequenceStatus.RUNNING
                    self._prepend_slot(seq_group)
                    seq.data.running_start_idx = seq.discard_start_idx
                    self.running.append(seq_group) # if no running_query, only swap in
                    is_prompt.append(False)
                else:
                    not_running.append(seq_group)
                seq.discard_start_idx += inflight_tokens
                seq.discard_length -= inflight_tokens
                
        self.resuming = delayed_resuming + not_running
        # schedule swapped group
        assert not self.swapped, 'use resuming group to resume swapped seqs now'
        
        ignored_seq_groups: List[SequenceGroup] = []
        scheduled_promtp = 0
        if not self.resuming and not preempted:
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            num_batched_tokens = sum(seq_group.get_seqs()[0].data.running_query_len for seq_group in self.running)
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue
                
                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break
                max_mul, _ = self._get_max_ragged_batch(self.running, blocks_to_swap_in, blocks_to_swap_out)
                if max_mul <= 0:
                    break
                inflight_tokens = min(num_prompt_tokens, max_mul)
                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + inflight_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                if not self.block_manager.can_allocate(seq_group, inflight_tokens):
                    # only do passive discard for Vulcan
                    if self.scheduler_config.api_policy != 'V':
                        break
                    else:
                        free_gpu = self.block_manager.get_num_free_gpu_blocks()
                        new_blocks = len(seq_group.get_seqs()[0].logical_token_blocks) 
                        if self.virtual_passive_discard(new_blocks - free_gpu):
                            self.passive_discard(new_blocks - free_gpu)
                        else:
                            break
                # If the sequence group cannot be allocated, stop.

                seq_group = self.waiting.pop(0)
                seq_group.get_seqs()[0].data.running_query_len = inflight_tokens
                self._allocate(seq_group)
                # insert to the front, to align with model input/sample output order
                self.running.insert(scheduled_promtp, seq_group)
                scheduled_promtp += 1
                is_prompt.insert(0, True)
                num_batched_tokens += inflight_tokens
                num_curr_seqs += num_new_seqs
        
        num_batched_tokens = sum(seq_group.get_seqs()[0].data.running_query_len for seq_group in self.running)
        self.total_tks += num_batched_tokens
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            is_prompt=is_prompt,
        )
        return scheduler_outputs

    def schedule(self, paused_swap_out: Dict[int, int] = None) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Change not fully resumed seq groups back to resuming status
        # Change fully resumed seqs back to running state
        self.debug_iter += 1
        # self.migrate_resumed_seq_groups()
        self.running.extend(self.delayed_running)
        self.delayed_running = []
        
        _, max_swap = self._get_max_ragged_batch([], [], [])
        self.free_swap_out_left = max_swap // 2
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        if self.scheduler_config.chunk_fill and self.scheduler_config.chunk_size > 0:
            scheduler_outputs = self._schedule_chunk_and_fill()
        else:
            scheduler_outputs = self._schedule()
        if paused_swap_out:
            scheduler_outputs.blocks_to_swap_out.update(paused_swap_out)
        
        paused_seqs: List[SequenceGroup] = [group for group, m in self.paused.values() if m is PreemptionMode.PRESERVE]
        paused_gpu_blocks = sum(len(self.block_manager.get_seq_blocks_by_device(group, Device.GPU)) for group in paused_seqs)
        
        recomp_tokens = sum(group.get_seqs()[0].get_len() for group in scheduler_outputs.scheduled_seq_groups if group.discarded)
        recomp_mem_blocks = sum(len(self.block_manager.get_seq_blocks_by_device(group, Device.GPU)) for group in scheduler_outputs.scheduled_seq_groups if group.discarded)
        num_non_free_GPU_blocks = self.block_manager.num_total_gpu_blocks - len(self.block_manager.gpu_allocator.free_blocks)
        num_not_involved_blocks = num_non_free_GPU_blocks - sum(len(self.block_manager.get_seq_blocks_by_device(group, Device.GPU)) for group in scheduler_outputs.scheduled_seq_groups)
        
        self.iter_history.append(IterStat(
            paused_gpu_blocks=paused_gpu_blocks,
            not_involved_gpu_blocks=num_not_involved_blocks,
            batch_size=scheduler_outputs.num_batched_tokens,
            recomp_token_size=recomp_tokens,
            recomp_mem_blocks=recomp_mem_blocks,
            swap_in_blocks=len(scheduler_outputs.blocks_to_swap_in),
            swap_out_blocks=len(scheduler_outputs.blocks_to_swap_out)
        ))
        
        # assert scheduler_outputs.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens, f'{scheduler_outputs.num_batched_tokens} > {self.scheduler_config.max_num_batched_tokens}'
        if scheduler_outputs.num_batched_tokens > self.scheduler_config.max_num_batched_tokens:
            logger.warning(f"Batched tokens ({scheduler_outputs.num_batched_tokens}) exceeds limit of {self.scheduler_config.max_num_batched_tokens}")
        # logger.info(f'num_seqs: {len(scheduler_outputs.scheduled_seq_groups)}, num_tokens: {scheduler_outputs.num_batched_tokens}')
        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group, prompt in zip(scheduler_outputs.scheduled_seq_groups, scheduler_outputs.is_prompt):
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished() and not seq_group.is_paused()
        ]
    
    def migrate_resumed_seq_groups(self) -> None:
        running, resuming = [], []
        for seq_group in self.running + self.resuming:
            seq = seq_group.get_seqs()[0]
            if seq.fully_resumsed():
                seq.status = SequenceStatus.RUNNING
                running.append(seq_group)
                seq.discard_start_idx = 0
                seq.swap_start_idx = 0
            else:
                seq.status = SequenceStatus.RESUMING
                resuming.append(seq_group)
        self.running = running
        self.resuming = resuming

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]
    
    def _prepend_slot(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            self.block_manager.prepend_slot(seq)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = PreemptionMode.RECOMPUTE,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            if self.block_manager.can_swap_out(seq_group):
                self._preempt_by_swap(seq_group, blocks_to_swap_out)
            else:
                self._preempt_by_recompute(seq_group)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs()
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            seq.data.clear_workload()
            seq.data.discard_start_idx = seq.data.discard_length = 0
            seq.data.swap_start_idx = seq.data.swap_length = 0
            seq.data.inflight_length = seq.data.get_len()
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        seq_group.discarded = True
        self.waiting.insert(0, seq_group)

    # this assumes no blocks are discarded
    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seq = seq_group.get_seqs()[0]
        assert seq.seq_id in self.block_manager.block_tables 
        assert len(self.block_manager.get_block_table(seq)) == self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length)
        self._swap_out(seq_group, blocks_to_swap_out)
        seq.data.clear_workload()
        seq.data.discard_start_idx = seq.data.discard_length = 0
        seq.data.swap_start_idx = 0
        seq.data.swap_length = len(self.block_manager.get_block_table(seq))
        self.swapped.append(seq_group)
        # self.resuming.append(seq_group)
        
    def _preempt_resuming(self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        still_resuming: List[SequenceGroup],
        preemption_mode: Optional[PreemptionMode] = PreemptionMode.RECOMPUTE,
    ) -> None:
        # If victim has no GPU blocks, do nothing
        if not self.block_manager.get_seq_blocks_by_device(seq_group, device=Device.GPU):
            still_resuming.append(seq_group)
            return
        if preemption_mode is PreemptionMode.RECOMPUTE:
            self.preempt_resuming_by_recompute(seq_group)
        else:
            if self.block_manager.can_swap_out(seq_group):
                self.preempt_resuming_by_swap(seq_group, blocks_to_swap_out)
                still_resuming.append(seq_group)
            else:
                self.preempt_resuming_by_recompute(seq_group)
    
    def preempt_resuming_by_recompute(self, seq_group: SequenceGroup) -> None:
        self._preempt_by_recompute(seq_group)
        for seq in seq_group.get_seqs():
            seq.swap_start_idx = 0
            seq.swap_length = 0
            seq.discard_start_idx = 0
            seq.discard_length = 0
    
    # Swap out remaining GPU blocks, increase the swap length only
    def preempt_resuming_by_swap(
        self, 
        seq_group: SequenceGroup, 
        blocks_to_swap_out: Dict[int, int]
    ) -> None:
        mapping, _ = self.block_manager.swap_out(seq_group, status=SequenceStatus.RESUMING)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RESUMING):
            seq.swap_length += len(mapping)
        

    # only do swap in, change status outside plz
    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
        max_swap_blocks: int = -1,
        status: SequenceStatus = SequenceStatus.SWAPPED,
    ) -> int:
        mapping = self.block_manager.swap_in(seq_group, max_swap_blocks, status)
        blocks_to_swap_in.update(mapping)
        return len(mapping)

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping, all_swapped = self.block_manager.swap_out(seq_group)
        assert all_swapped
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
