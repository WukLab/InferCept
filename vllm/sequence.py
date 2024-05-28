"""Sequence and its related classes."""
import copy
import enum
from typing import Dict, List, Optional, Union

from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams
from vllm.utils import invalidate_access
@invalidate_access(["RESUMING"])
class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    PAUSED_API = enum.auto()
    RESUMED_API = enum.auto()
    RESUMING = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]
    
    @staticmethod
    def is_paused(status: "SequenceStatus") -> bool:
        return status == SequenceStatus.PAUSED_API

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason

@invalidate_access(["logical_query_len", "running_query_len", "running_start_idx"])
class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
        logical_query_len: The length of queries need to be calculated to 
                            catch up with the sequence
        running_query_len: The length of queries that are forwarded in the current iteration
        running_start_idx: The index of the first token forwarded in the current iteration
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0
        self.logical_query_len = len(prompt_token_ids)
        self.running_query_len = 0
        self.running_start_idx = 0
        self.generation_counter = 0
        self.seen_tokens = 0
        
        """sequence token format
        |G|<   discard   >|G|G|<     swap    >|G|<  in-flight  >|
        |D|d_start + d_len|P|S|s_start + s_len|P|     i_len     |
        """
        # In number of tokens
        self.discard_start_idx = 0
        self.discard_length = 0
        # In number of blocks
        self.swap_start_idx = 0
        self.swap_length = 0
        # In number of tokens, means the number of tokens that are not part of resuming
        self.inflight_length = len(prompt_token_ids)
        # Workload
        self.resume_discard_tokens = 0
        self.resume_swap_blocks = 0
        self.running_inflight_tokens = 0
    
    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob
        self.inflight_length += 1

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def is_generating(self) -> bool:
        return self.inflight_length == self.running_inflight_tokens
    
    def update_seen_tokens(self):
        # self.seen_tokens = max(self.seen_tokens, self.get_len() - self.inflight_length + self.running_inflight_tokens)
        self.seen_tokens = self.get_len()
    
    def num_required_blocks(self, inflight_tokens: int, block_size: int) -> int:
        toks = self.get_len() - self.logical_query_len + inflight_tokens
        return (toks + block_size - 1) // block_size
    
    def num_required_blocks_resume_discard(self, block_size: int) -> int:
        resumed = (self.discard_start_idx + block_size - 1) // block_size
        after_resumed = (self.discard_start_idx + self.resume_discard_tokens + block_size - 1) // block_size
        return after_resumed - resumed

    def num_required_blocks_resume_swap(self) -> int:
        return self.resume_swap_blocks

    def num_required_blocks_inflight(self, block_size: int) -> int:
        after_resume_blocks = (self.get_len() - self.inflight_length + block_size - 1) // block_size
        total_blocks = (self.get_len() - self.inflight_length + self.running_inflight_tokens + block_size - 1) // block_size
        return total_blocks - after_resume_blocks
    
    def populate_workload(self, resume_discard_tokens: int, resume_swap_blocks: int, running_inflight_tokens: int):
        self.resume_discard_tokens = resume_discard_tokens
        self.resume_swap_blocks = resume_swap_blocks
        self.running_inflight_tokens = running_inflight_tokens
    
    def post_forward(self):
        self.discard_length -= self.resume_discard_tokens
        self.swap_start_idx += self.resume_swap_blocks
        self.swap_length -= self.resume_swap_blocks
        self.inflight_length -= self.running_inflight_tokens
        self.clear_workload()
    
    def clear_workload(self):
        self.resume_discard_tokens = 0
        self.resume_swap_blocks = 0
        self.running_inflight_tokens = 0

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: List[Dict[int, float]] = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None
        # In number of blocks
        self.discard_idx = 0

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor:cursor +
                                               num_empty_slots])
            cursor += num_empty_slots

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids
    
    def get_num_required_blocks_inflight(self, inflight_tokens: int) -> int:
        return self.data.num_required_blocks(inflight_tokens, self.block_size)
    
    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 0.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # Note: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)
    
    def is_paused(self) -> bool:
        return SequenceStatus.is_paused(self.status)
    
    def fully_resumsed(self) -> bool:
        return self.discard_length == 0 and self.swap_length == 0

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq
    
    #NOTE: Prepare workload first before using the following functions
    @property
    def num_new_blocks_for_resume_discard(self) -> int:
        return self.data.num_required_blocks_resume_discard(self.block_size)
    @property
    def num_new_blocks_for_resume_swap(self) -> int:
        return self.data.num_required_blocks_resume_swap()
    @property
    def num_new_blocks_for_inflight(self) -> int:
        return self.data.num_required_blocks_inflight(self.block_size)

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.discarded = False

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            # For beam search, maximally there will always be `best_of` beam
            # candidates running in the future.
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_seqs():
                # At prompt stage, the sequence group is not yet filled up
                # and only have one sequence running. However, in the
                # generation stage, we will have `best_of` sequences running.
                return self.sampling_params.best_of
            # At sampling stages, return the number of actual sequences
            # that are not finished yet.
            return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [
                seq for seq in self.seqs_dict.values() if seq.status == status
            ]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())
    
    def is_paused(self) -> bool:
        return any(seq.is_paused() for seq in self.get_seqs())

    def get_total_discarded(self) -> int:
        return sum(s.discard_idx for s in self.get_unfinished_seqs())
    
    def api_remaining_time(self, now: float):
        return self.sampling_params.api_exec_time - (now - self.sampling_params.api_call_time)
    
    # def __lt__(self, other):
    #     if self.get_total_discarded() != other.get_total_discarded():
    #         return self.get_total_discarded() < other.get_total_discarded()
    #     return  self.sampling_params.api_exec_time >= other.sampling_params.api_exec_time

    def __eq__(self, other) -> bool:
        return self.request_id == other.request_id
        # return self.api_remaining_time() == other.api_remaining_time()
        # return self.sampling_params.api_exec_time == other.sampling_params.api_exec_time and \
        #     self.get_total_discarded() == other.get_total_discarded()
    
    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs_dict)})")


class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
        only_swap: bool = False,
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
        self.only_swap = only_swap


class SequenceOutputs:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutputs(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}), "
                f"logprobs={self.logprobs}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutputs):
            raise NotImplementedError()
        return (self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)


# For each sequence group, we generate a list of SequenceOutputs object,
# each of which contains one possible candidate for the next token.
SamplerOutput = List[List[SequenceOutputs]]
