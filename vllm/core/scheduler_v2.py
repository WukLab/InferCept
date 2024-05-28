import enum
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import SchedulerOutputs, PreemptionMode
from vllm.core.block_manager import BlockSpaceManager, Device
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from vllm.core.solver import Solver
from heapq import *
from collections import deque
from functools import total_ordering
import math

logger = init_logger(__name__)

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

class SchedulerV2:
    """Scheduler that supports chunk-fill
       also provides new abstraction for seq-group management
    """
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
        # free-gpu-blocks, batched token, discard, resume
        self.history: List[Tuple[int, int, int, int]] = []
        self.paused_gpu_blocks: List[Tuple[int, int]] = []
        self.iter_history: List[IterStat] = []
        
        self.total_batched_tokens = 0
        print(f'swap limit const: {self.scheduler_config.swap_limit_constant}')
        if self.scheduler_config.api_policy in ['G', 'V']:
            assert self.scheduler_config.chunk_fill, 'V/G policy only works with chunk fill to enforce limit'
            
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def get_tokens_have_seen(self) -> int:
        groups = self.running + self.swapped + self.waiting + [group for group, _ in self.paused.values()]
        return sum(seq.data.seen_tokens for seq_group in groups for seq in seq_group.get_seqs())

    def pause_by_recompute(self, seq_group: SequenceGroup):
        for seq in seq_group.get_seqs():
            seq.data.clear_workload()
            seq.data.discard_start_idx = seq.data.discard_length = 0
            seq.data.swap_start_idx = seq.data.swap_length = 0
            seq.data.inflight_length = seq.data.get_len()
            self.block_manager.free(seq) 
        seq_group.discarded = True
        self.paused[seq_group.request_id] = (seq_group, PreemptionMode.RECOMPUTE)
    
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
        elif self.scheduler_config.api_policy == 'H-S':
            if seq_group.sampling_params.api_exec_time >= 4:
                mode = PreemptionMode.SWAP
            else:
                mode = PreemptionMode.PRESERVE
        elif self.scheduler_config.api_policy == 'H-D':
            if seq_group.sampling_params.api_exec_time >= 7:
                mode = PreemptionMode.RECOMPUTE
            else:
                mode = PreemptionMode.PRESERVE
        elif self.scheduler_config.api_policy == 'H-B':
            if seq_group.sampling_params.api_exec_time < 4:
                mode = PreemptionMode.PRESERVE
            elif seq_group.sampling_params.api_exec_time < 7:
                mode = PreemptionMode.SWAP
            else:
                mode = PreemptionMode.RECOMPUTE
        elif self.scheduler_config.api_policy == 'G':
            # This state will be changed at the next scheduling iter
            # If a short api, resuming in Greedy mode will turn it into a running seq
            mode = PreemptionMode.PRESERVE
        elif self.scheduler_config.api_policy == 'D':
            mode = PreemptionMode.RECOMPUTE
        elif self.scheduler_config.api_policy == 'S':
            mode = PreemptionMode.SWAP
        elif self.scheduler_config.api_policy == 'P':
            mode = PreemptionMode.PRESERVE
        elif self.scheduler_config.api_policy == 'V':
            mode = PreemptionMode.PRESERVE 
        else:
            raise ValueError(f"Invalid API policy {self.scheduler_config.api_policy}")
        # Deal with it
        if mode is PreemptionMode.PRESERVE:
            self.paused[seq_group.request_id] = (seq_group, PreemptionMode.PRESERVE)
        if mode is PreemptionMode.RECOMPUTE:
            self.pause_by_recompute(seq_group)
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
                self.paused[seq_group.request_id] = (seq_group, PreemptionMode.PRESERVE)
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
            self.waiting.insert(0, seq_group)
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
        return self.waiting or self.running or self.swapped or self.paused

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + len(self.paused)
    
    def get_num_batched_tokens(self, requests: List[SequenceGroup]) -> int:
        num_batched_tokens = sum(seq_group.get_seqs()[0].data.resume_discard_tokens for seq_group in requests)
        num_batched_tokens += sum(seq_group.get_seqs()[0].data.running_inflight_tokens for seq_group in requests)
        return num_batched_tokens
    
    def passive_discard_swap_only(
        self, 
        required_blocks: int, 
        victim_group: List[SequenceGroup],
        blocks_to_swap_out: Dict[int, int],
        preempted: List[SequenceGroup],
    ) -> int:
        victim = victim_group[-1]
        gpu_space = len(self.block_manager.get_seq_blocks_by_device(victim, Device.GPU))
        if not gpu_space:
            return
        seq = victim.get_seqs()[0]
        num_swap_blocks = min(gpu_space, required_blocks)
        self.block_manager.swap_out_from_back(seq, num_swap_blocks, blocks_to_swap_out)
        if not seq.data.swap_length:
            seq.data.swap_start_idx = self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length) - num_swap_blocks
        else:
            seq.data.swap_start_idx -= num_swap_blocks
        seq.data.swap_length += num_swap_blocks
        
        # if all swapped out, add to the correct group
        if not self.block_manager.get_seq_blocks_by_device(victim, Device.GPU):
            assert seq.data.swap_start_idx == 0
            assert seq.data.swap_length == self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length)
            if victim.request_id not in self.paused:
                self._preempt_by_swap(victim, blocks_to_swap_out)
            else:
                seq.data.clear_workload()
                seq.data.discard_start_idx = seq.data.discard_length = 0
                seq.data.swap_start_idx = 0
                seq.data.swap_length = len(self.block_manager.get_block_table(seq))
                self.paused[victim.request_id] = (victim, PreemptionMode.SWAP)
            victim_group.pop(-1)
            preempted.append(victim)
        return num_swap_blocks
    
    # NOTE: victim_group is sorted in api remaining time
    #       no api time is also considered as sorted
    def passive_discard_by_order(
        self, 
        required_blocks: int, 
        victim_group: List[SequenceGroup],
        blocks_to_swap_out: Dict[int, int],
        preempted: List[SequenceGroup],
        now: float, 
    ) -> int:
        total_swap_blocks = 0
        swapped_paused_group = [g for g, m in self.paused.values() if m is PreemptionMode.SWAP]
        # swapped_discard_list = self.policy.sort_by_priority(now, swapped_paused_group + self.swapped)
        swapped_discard_list = sorted(swapped_paused_group, key = lambda k: k.api_remaining_time(now))
        swapped_discard_list = self.swapped + swapped_discard_list
        while required_blocks > 0:
            if not victim_group:
                break
            victim = victim_group[-1]
            gpu_space = len(self.block_manager.get_seq_blocks_by_device(victim, Device.GPU))
            if not gpu_space:
                raise ValueError('Fully preempted seqs should not stay in the victim group')
            seq = victim.get_seqs()[0]
            num_swap_blocks = min(gpu_space, required_blocks)
            # If cpu mem is full, passive dicard longest one
            while self.block_manager.cpu_allocator.get_num_free_blocks() < num_swap_blocks:
                if not swapped_discard_list:
                    break
                discard_victim: SequenceGroup = swapped_discard_list.pop(-1)
                if discard_victim.request_id in self.paused:
                    self.pause_by_recompute(discard_victim)
                else:
                    self.swapped.remove(discard_victim)
                    self._preempt_by_recompute(discard_victim)
                
            # discard current if still not enought CPU mem
            # TODO: discard gpu blocks only
            if self.block_manager.cpu_allocator.get_num_free_blocks() < num_swap_blocks:
                required_blocks -= gpu_space
                if victim.request_id not in self.paused:
                    self._preempt_by_recompute(victim)
                else:
                    self.pause_by_recompute(victim)
                victim_group.pop(-1)
                preempted.append(victim)
            else:
                self.block_manager.swap_out_from_back(seq, num_swap_blocks, blocks_to_swap_out)
                required_blocks -= num_swap_blocks
                if not seq.data.swap_length:
                    seq.data.swap_start_idx = self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length) - num_swap_blocks
                else:
                    seq.data.swap_start_idx -= num_swap_blocks
                seq.data.swap_length += num_swap_blocks
                total_swap_blocks += num_swap_blocks
                # if all swapped out, add to the correct group
                if not self.block_manager.get_seq_blocks_by_device(victim, Device.GPU):
                    assert seq.data.swap_start_idx == 0
                    assert seq.data.swap_length == self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length)
                    if victim.request_id not in self.paused:
                        self._preempt_by_swap(victim, blocks_to_swap_out)
                    else:
                        seq.data.clear_workload()
                        seq.data.discard_start_idx = seq.data.discard_length = 0
                        seq.data.swap_start_idx = 0
                        seq.data.swap_length = len(self.block_manager.get_block_table(seq))
                        self.paused[victim.request_id] = (victim, PreemptionMode.SWAP)
                    victim_group.pop(-1)
                    preempted.append(victim)
        return total_swap_blocks
            
    def passive_discard(
        self, 
        required_blocks: int, 
        victim_group: List[SequenceGroup],
        blocks_to_swap_out: Dict[int, int],
        preempted: List[SequenceGroup],
        now: float,
    ) -> int:
        return self.passive_discard_by_order(required_blocks, victim_group, blocks_to_swap_out, preempted, now)

    # TODO: Consider discard as well, currently only swap
    def passive_discard_complex(
        self, 
        required_blocks: int, 
        victim_group: List[SequenceGroup],
        blocks_to_swap_out: Dict[int, int],
        preempted: List[SequenceGroup],
    ) -> int:
        return self.passive_discard_swap_only(required_blocks, victim_group, blocks_to_swap_out, preempted)
        """memory format:
        |1|               |2|3|               |4|
        |G|<   discard   >|G|G|<     swap    >|G|<  in-flight  >|
        |D|d_start + d_len|P|S|s_start + s_len|P|     i_len     |
        eviction order: 4 -> (3/2) -> 1 -> (2/3), by S -> S -> D -> D
        """
        victim = victim_group[-1]
        gpu_space = self.block_manager.get_seq_blocks_by_device(victim, Device.GPU)
        if not gpu_space:
            return
        seq = victim.get_seqs()[0]
        # Preempt part 4:
        end_4 = self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length)
        if seq.data.swap_length:
            start_4 = seq.data.swap_start_idx + seq.data.swap_length
        elif seq.data.discard_length:
            start_4 = self.block_manager.token_2_block(seq.data.discard_start_idx + seq.data.discard_length)
        else:
            start_4 = 0
        num_part_4 = min(end_4 - start_4, required_blocks)
        self.block_manager.swap_out_from_back(seq, num_part_4, blocks_to_swap_out)
        required_blocks -= num_part_4
        if not seq.data.swap_length:
            seq.data.swap_start_idx = self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length) - num_part_4
        seq.data.swap_length += num_part_4
        
        # Preempt part 3/2:
        if seq.data.swap_length:
            end_3 = seq.data.swap_start_idx
        else:
            end_3 = self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length)
        if seq.data.discard_length:
            start_2 = self.block_manager.token_2_block(seq.data.discard_start_idx + seq.data.discard_length)
        else:
            start_2 = 0
        num_part_3 = min(end_3-start_2, required_blocks)
        self.block_manager.swap_out_from_back(seq, num_part_3, blocks_to_swap_out)
        required_blocks -= num_part_3
        if not seq.data.swap_length:
            seq.data.swap_start_idx = self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length) - num_part_3
        else:
            seq.data.swap_start_idx -= num_part_3
        seq.data.swap_length += num_part_3
        
        assert seq.data.swap_length == len(self.block_manager.get_seq_blocks_by_device(victim, Device.CPU))
        if seq.data.swap_length:
            assert seq.data.swap_start_idx == self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length) - seq.data.swap_length
        
        if seq.seq_id not in self.block_manager.block_tables:
            assert seq.data.discard_start_idx == 0
            assert seq.data.discard_length == seq.get_len()
            if victim.request_id not in self.paused:
                self._preempt_by_recompute(victim)
            else:
                seq.data.clear_workload()
                seq.data.discard_start_idx = seq.data.discard_length = 0
                seq.data.swap_start_idx = seq.data.swap_length = 0
                seq.data.inflight_length = seq.data.get_len()
                self.block_manager.free(seq) 
                self.paused[victim.request_id] = (victim, PreemptionMode.RECOMPUTE)
            victim_group.pop(-1)
            preempted.append(victim)
        if not self.block_manager.get_seq_blocks_by_device(victim, Device.GPU):
            assert seq.data.swap_start_idx == 0
            assert seq.data.swap_length == self.block_manager.token_2_block(seq.data.get_len() - seq.data.inflight_length)
            if victim.request_id not in self.paused:
                self._preempt_by_swap(victim, blocks_to_swap_out)
            else:
                seq.data.clear_workload()
                seq.data.discard_start_idx = seq.data.discard_length = 0
                seq.data.swap_start_idx = 0
                seq.data.swap_length = len(self.block_manager.get_block_table(seq))
                self.paused[victim.request_id] = (victim, PreemptionMode.SWAP)
            victim_group.pop(-1)
            preempted.append(victim)
        return 
    
    def _schedule(
        self,
        blocks_to_swap_out: Dict[int, int],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        schedule_targets: List[SequenceGroup],
        preemption_targets: List[SequenceGroup],
        preempt_if_need: bool,
        self_preempt: bool,
        max_sequences: int,
        stop_if_batch_limit: bool,
        stop_if_no_swap: bool,
        now: float,
    ) -> Tuple[List[SequenceGroup], List[SequenceGroup], List[SequenceGroup], List[SequenceGroup]]:
        """
        This method meant to provide a unified schedule api for all groups
        NOTE: 1. schedule_targets are always preemptable unless not preempt_if_need or not self_preempt.
              2. schedule_targets should be sorted before calling this function.
              3. scheduled will be peeled off from schedule_targets
              
        If not preempt_if_need, will stop scheduling once memory is full, not even try to preempt itself
        If stop_if_batch_limit, will stop scheduling once batch limit is reached
        If self_preempt, will preempt itself or the group it resides if needed
        if stop_if_no_swap, will stop scheduling once requred blocks exceeds allowed swap out
        
        Return: scheduled, preempted, delayed, ignored
        """
        num_batched_tokens = 0
        num_batched_seqs = 0
        scheduled: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        ignored: List[SequenceGroup] = []
        # Not preempted neither scheduled
        delayed: List[SequenceGroup] = []
        
        if preemption_targets:
            assert preempt_if_need
        
        counter = 0
        while schedule_targets:
            counter += 1
            if counter > 100000:
                pass
            seq_group = schedule_targets.pop(0)
            seq = seq_group.get_seqs()[0]
            if seq.get_len() > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({seq.get_len()} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored.append(seq_group)
                continue
            # Check batch limit
            if num_batched_seqs + seq_group.get_max_num_running_seqs() > max_sequences:
                delayed.append(seq_group)
                break
            
            # Chunk swap workload
            # TODO: jointly consider swap in
            n_swap_blocks = min(self.max_swap_in_blocks, seq.data.swap_length)
            # Chunk computation workload
            if seq.data.swap_length == n_swap_blocks:
                max_tokens_to_schedule = self.batch_max_tokens - num_batched_tokens
                n_inflight_tokens = min(max_tokens_to_schedule, seq.data.inflight_length)
            else:
                # If resuming not complete, do not schedule inflight tokens
                n_inflight_tokens = 0
            
            if n_inflight_tokens <= 0 and n_swap_blocks <= 0:
                # Meet batch limit
                delayed.append(seq_group)
                if stop_if_batch_limit:
                    break
                continue
            
            # Get memory requirement
            assert seq.data.discard_length == 0
            seq.data.populate_workload(
                seq.data.discard_length, n_swap_blocks, n_inflight_tokens)
            new_blocks = self.block_manager.get_memory_requirement(seq_group)
            if not self.block_manager.can_fulfill_blocks(new_blocks):
                if not preempt_if_need:
                    delayed.append(seq_group)
                    break
                if not preemption_targets and not self_preempt:
                    delayed.append(seq_group)
                    break
                if not self_preempt:
                    if new_blocks - self.block_manager.gpu_allocator.get_num_free_blocks() > self.max_swap_out_blocks and stop_if_no_swap:
                        delayed.append(seq_group)
                        break
            
            # Preemption
            while not self.block_manager.can_fulfill_blocks(new_blocks):
                num_required_blocks = new_blocks - self.block_manager.gpu_allocator.get_num_free_blocks()
                if num_required_blocks > self.max_swap_out_blocks:
                    delayed.append(seq_group)
                    break
                if preemption_targets:
                    n_sout = self.passive_discard(num_required_blocks, preemption_targets, blocks_to_swap_out, preempted, now)
                    self.max_swap_out_blocks -= n_sout
                elif not self_preempt:
                    delayed.append(seq_group)
                    break
                elif delayed:
                    n_sout = self.passive_discard(num_required_blocks, delayed, blocks_to_swap_out, preempted, now)
                    self.max_swap_out_blocks -= n_sout
                elif schedule_targets:
                    n_sout = self.passive_discard(num_required_blocks, schedule_targets, blocks_to_swap_out, preempted, now)
                    self.max_swap_out_blocks -= n_sout
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self.passive_discard(len(self.block_manager.get_seq_blocks_by_device(seq_group, Device.GPU)), [seq_group], blocks_to_swap_out, preempted, now)
                    break
            else:
                # Append new slots to the sequence group.
                num_batched_tokens += seq.data.running_inflight_tokens
                self.block_manager.allocate(seq_group, blocks_to_swap_in)
                seq.status = SequenceStatus.RUNNING
                scheduled.append(seq_group)
                self.max_swap_in_blocks -= n_swap_blocks
                
        self.batch_max_tokens -= num_batched_tokens
        for seq_group in delayed:
            seq_group.get_seqs()[0].data.clear_workload()
        return scheduled, preempted, delayed, ignored

    def _active_discard(self, now):
        if not self.waiting:
            # logger.info('discard swapped')
            if self.swapped:
                self.swapped = self.policy.sort_by_priority(now, self.swapped)
                victim = self.swapped.pop(0)
                self._preempt_by_recompute(victim)
    
    # def _schedule_chunk_by_order(
    #     self,
    #     blocks_to_swap_out: Dict[int, int],
    #     blocks_to_swap_in: Dict[int, int],
    #     blocks_to_copy: Dict[int, List[int]],
    # ) -> SchedulerOutputs:
    #     now = time.monotonic()
    #     self._active_discard(now)
    #     scheduled: List[SequenceGroup] = []
    #     ignored: List[SequenceGroup] = []
    #     is_prompt: List[bool] = []
    #     paused_preemption_target = sorted([seq_group for seq_group, mode in self.paused.values() if mode is PreemptionMode.PRESERVE], key=lambda x: x.api_remaining_time(now))
    #     max_num_seqs = self.scheduler_config.max_num_seqs
        
    #     schedule_target = self.policy.sort_by_priority(now, self.running + self.swapped)
        
    def discard_waste(self, num_blocks: int):
        running_batch = sum(seq_group.get_seqs()[0].data.inflight_length for seq_group in self.running)
        running_blocks = sum(len(self.block_manager.get_seq_blocks_by_device(group, Device.GPU)) for group in self.running)
                
        c_h = max(384 - running_batch, 1)
        n = max((self.block_manager.block_size * num_blocks + c_h - 1) // c_h - 1, 0)
        
        a = 0.0279 # replace f_ch a
        c = 15.4 # replace f_ch c
        f_ch = (a * c_h) / 1000
        f_s = (a * 384 + c) / 1000
        
        w_d = f_s * (1+n) * n / 2 * c_h + f_ch * n * running_blocks * self.block_manager.block_size
        last_resume_toks = (self.block_manager.block_size * num_blocks) % c_h
        f_last_resume = (a * max(0, last_resume_toks)) / 1000
        w_d += f_last_resume * (running_blocks * self.block_manager.block_size + last_resume_toks)
        return w_d
    
    def _schedule_chunk_and_fill(
        self,
        blocks_to_swap_out: Dict[int, int],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SchedulerOutputs:
        now = time.monotonic()
        self._active_discard(now)
        scheduled: List[SequenceGroup] = []
        ignored: List[SequenceGroup] = []
        is_prompt: List[bool] = []

        # get number of blocks in running group
        num_running_seqs = 0
        running_blocks = 0
        for sg in self.running:
            seq = sg.get_seqs()[0]
            if seq.seq_id in self.block_manager.block_tables:
                running_blocks += len(self.block_manager.get_block_table(seq))
                num_running_seqs += 1

        paused_preemption_candidates = {}
        max_waste = -1
        running_batch = sum(seq_group.get_seqs()[0].data.inflight_length for seq_group in self.running)
        for seq_group, m in self.paused.values():
            if m is not PreemptionMode.PRESERVE:
                continue 
            seq_blocks = len(self.block_manager.get_seq_blocks_by_device(seq_group, Device.GPU))
            # preserve waste = remaining api time * sequence blocks
            w_p = seq_group.api_remaining_time(now) * seq_blocks * self.block_manager.block_size

            w_d = self.discard_waste(seq_blocks)

            if w_d < w_p:
                # recompute
                paused_preemption_candidates[seq_group.request_id] = (w_d, PreemptionMode.RECOMPUTE)
                max_waste = max(max_waste, w_d)
            # logger.info(f'w_d: {w_d}, w_p: {w_p}')
            
        paused_preemption_target = []
        for rid, waste in paused_preemption_candidates.items():
            waste_term, mode = waste
            if waste_term < max_waste:
                if mode == PreemptionMode.RECOMPUTE:
                    self.pause_by_recompute(self.paused[rid][0])
            else:
                paused_preemption_target.append(self.paused[rid][0])
        
        max_num_seqs = self.scheduler_config.max_num_seqs
        self.running = self.policy.sort_by_priority(now, self.running)
        scheduled_running, preempted, delayed, ignored_running = self._schedule(
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            schedule_targets=self.running,
            preemption_targets=paused_preemption_target,
            preempt_if_need=True,
            self_preempt=True,
            max_sequences=max_num_seqs,
            stop_if_batch_limit=False,
            stop_if_no_swap=False,
            now=now,
        )
        self.running.extend(scheduled_running + delayed)
        scheduled.extend(scheduled_running)
        ignored.extend(ignored_running)
        is_prompt.extend([False] * len(scheduled_running))
        
        max_num_seqs = self.scheduler_config.max_num_seqs - len(scheduled)
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        scheduled_swap, _, delayed, ignored_swap = self._schedule(
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            schedule_targets=self.swapped,
            preemption_targets=paused_preemption_target,
            preempt_if_need=True,
            self_preempt=False,
            max_sequences=max_num_seqs,
            stop_if_batch_limit=False,
            stop_if_no_swap=True,
            now=now,
        )
        self.running.extend(scheduled_swap)
        self.swapped.extend(delayed)
        scheduled.extend(scheduled_swap)
        ignored.extend(ignored_swap)
        is_prompt.extend([False] * len(scheduled_swap))
        
        # if not preempted and not self.swapped:
        max_num_seqs = self.scheduler_config.max_num_seqs - len(scheduled)
        self.waiting = self.policy.sort_by_priority(now, self.waiting)
        scheduled_prompt, _, delayed, ignored_prompt = self._schedule(
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            schedule_targets=self.waiting,
            preemption_targets=paused_preemption_target,
            preempt_if_need=True,
            self_preempt=False,
            max_sequences=max_num_seqs,
            stop_if_batch_limit=True,
            stop_if_no_swap=True,
            now=now,
        )
        self.running.extend(scheduled_prompt)
        delayed.extend(self.waiting)
        self.waiting = delayed
        scheduled = scheduled_prompt + scheduled
        ignored.extend(ignored_prompt)
        is_prompt = [True] * len(scheduled_prompt) + is_prompt
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=scheduled,
            prompt_run=False,
            num_batched_tokens=self.get_num_batched_tokens(scheduled),
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored,
            is_prompt=is_prompt,
        )
        return scheduler_outputs
    
    def _set_max_ragged_batch(self, 
                              initial_free_toks: int):
        paused_seqs: List[SequenceGroup] = [group for group, m in self.paused.values() if m is PreemptionMode.PRESERVE]
        num_tokens_paused = sum(len(self.block_manager.get_seq_blocks_by_device(group, Device.GPU)) * self.block_manager.block_size for group in paused_seqs)
        # mlimit = max(min(self.scheduler_config.max_num_batched_tokens, num_tokens_paused + initial_free_toks - 1000), 0)
        
        a = -0.3297119141 # replace t_sin
        b = 93.994140625 # replace t_offset
        c = 1
        mlimit = 384
        t_sin = c * (a * mlimit + b) + initial_free_toks / 2
        t_sout = c * ((1+a) * mlimit + b) - initial_free_toks / 2
        self.batch_max_tokens = mlimit
        self.max_swap_in_blocks = max(0, t_sin // self.block_manager.block_size)
        self.max_swap_out_blocks = max(0, t_sout // self.block_manager.block_size)
        # logger.info(f'{mlimit}, {t_sout}, {t_sin}')
    
    def _if_repeat(self, group_list: List[SequenceGroup]) -> bool:
        repeat = set()
        for group in group_list:
            if group.request_id in repeat:
                return True
            repeat.add(group.request_id)
        return False
    
    def schedule(self, 
                 paused_swap_out: Dict[int, int] = None
            ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        free_gpu_tokens = self.block_manager.get_num_free_gpu_blocks() * self.block_manager.block_size
        self._set_max_ragged_batch(free_gpu_tokens)
        max_tokens = self.batch_max_tokens
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        if paused_swap_out:
            blocks_to_swap_out.update(paused_swap_out)
            
        scheduler_outputs = self._schedule_chunk_and_fill(blocks_to_swap_out, blocks_to_swap_in, blocks_to_copy)
        assert scheduler_outputs.num_batched_tokens <= max_tokens, f"Batched tokens ({scheduler_outputs.num_batched_tokens}) exceeds limit of {max_tokens}"
        
        # logger.info(f"Scheduler: {scheduler_outputs.num_batched_tokens} batched tokens, {len(scheduler_outputs.scheduled_seq_groups)} seq groups, {len(blocks_to_swap_in)} blocks to swap in, {len(blocks_to_swap_out)} blocks to swap out, {len(blocks_to_copy)} blocks to copy")
        
        paused_seqs: List[SequenceGroup] = [group for group, m in self.paused.values() if m is PreemptionMode.PRESERVE]
        paused_gpu_blocks = sum(len(self.block_manager.get_seq_blocks_by_device(group, Device.GPU)) for group in paused_seqs)

        recomp_tokens = sum(group.get_seqs()[0].data.running_inflight_tokens for group in scheduler_outputs.scheduled_seq_groups if group.discarded)
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
        
        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group, prompt in zip(scheduler_outputs.scheduled_seq_groups, scheduler_outputs.is_prompt):
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            sdata = seq_group.get_seqs()[0].data
            only_swap = sdata.running_inflight_tokens + sdata.resume_discard_tokens == 0
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                only_swap=only_swap,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        self.history.append((free_gpu_tokens, scheduler_outputs.num_batched_tokens, len(blocks_to_swap_out) * self.block_manager.block_size, len(blocks_to_swap_in) * self.block_manager.block_size))
        # logger.info(self.history[-1])
        self.total_tks += scheduler_outputs.num_batched_tokens
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
        preemption_mode: Optional[PreemptionMode] = PreemptionMode.SWAP,
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
        # self.waiting.insert(0, seq_group)
        self.waiting.append(seq_group)

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
