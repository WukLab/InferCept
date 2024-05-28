"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}
    
    def token_2_block(self, tok: int) -> int:
        return (tok + self.block_size - 1) // self.block_size

    def can_allocate(self, seq_group: SequenceGroup, inflight_tokens: int = -1) -> bool:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs()[0]
        if inflight_tokens == -1:
            num_required_blocks = len(seq.logical_token_blocks)
        else:
            num_required_blocks = self.token_2_block(inflight_tokens)
        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        return (num_free_gpu_blocks - num_required_blocks >=
                self.watermark_blocks)
    
    # NOTE: set workload before calling this function
    #       this should be the unifined interface for querying memory needs
    def get_memory_requirement(self, seq_group: SequenceGroup) -> int:
        seq = seq_group.get_seqs()[0]
        n_required_blocks = 0
        """Get memory requirement for resuming discarded tokens"""
        if seq.data.resume_discard_tokens:
            n_required_blocks += seq.num_new_blocks_for_resume_discard
        """Get memory requirement for resuming swapped tokens"""
        if seq.data.resume_swap_blocks:
            n_required_blocks += seq.num_new_blocks_for_resume_swap
        """Get memory requirement for appending inflight tokens"""
        if seq.data.running_inflight_tokens:
            assert seq.data.discard_length - seq.data.resume_discard_tokens == 0
            assert seq.data.swap_length - seq.data.resume_swap_blocks == 0
            n_required_blocks += seq.num_new_blocks_for_inflight
        return n_required_blocks
    
    # NOTE: populate workload before calling this function
    # TODO: currently assume all or nothing
    def allocate(self, seq_group: SequenceGroup, blocks_to_swap_in: Dict[int, int]) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs()[0]
        if seq.seq_id not in self.block_tables:
            self.block_tables[seq.seq_id] = []   
        block_table: BlockTable = self.block_tables[seq.seq_id]
        
        """Get memory requirement for resuming discarded tokens"""
        if seq.data.resume_discard_tokens:
            resumed_discard_blocks = self.token_2_block(seq.data.discard_start_idx)
            for i in range(seq.num_new_blocks_for_resume_discard):
                block = self.gpu_allocator.allocate()
                block_table.insert(resumed_discard_blocks + i, block)
                
        """Get memory requirement for resuming swapped tokens"""
        if seq.data.resume_swap_blocks:
            n_swap_in = seq.num_new_blocks_for_resume_swap
            mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            for idx, block in enumerate(block_table):
                if block.device == Device.GPU or n_swap_in == 0:
                    continue
                if block in mapping:
                    mapping[block].ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[block] = gpu_block
                    block_table[idx] = gpu_block
                    n_swap_in -= 1
                self.cpu_allocator.free(block)
            block_number_mapping = {
                cpu_block.block_number: gpu_block.block_number
                for cpu_block, gpu_block in mapping.items()
            }
            blocks_to_swap_in.update(block_number_mapping)
            
        """Get memory requirement for appending inflight tokens"""
        if seq.data.running_inflight_tokens:
            assert seq.data.discard_length - seq.data.resume_discard_tokens == 0
            assert seq.data.swap_length - seq.data.resume_swap_blocks == 0
            #TODO: if n > 1, add copy back
            if self.block_sliding_window:
                raise NotImplementedError("sliding window att not considered")
            for _ in range(seq.num_new_blocks_for_inflight):
                block = self.gpu_allocator.allocate()
                block.ref_count = seq_group.num_seqs()
                block_table.append(block)

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks
    
    # Calculate the new block that need to be allocated
    # NOTE: inflight tokens represents the number of tokens that are forwarded to the model
    def required_new_blocks(self, seq_group: SequenceGroup, inflight_tokens: int = -1) -> int:
        seqs = seq_group.get_seqs()
        assert len(seqs) == 1
        r = 0
        for seq in seqs:
            physical_blocks = len(self.block_tables[seq.seq_id])
            if inflight_tokens == -1:
                logical_blocks = len(seq.logical_token_blocks)
            else:
                logical_blocks = seq.get_num_required_blocks_inflight(inflight_tokens)
            r += max(0, logical_blocks - physical_blocks)
        return r

    # Determine whether required number of new blocks can be fulfilled
    def can_fulfill_blocks(self, r: int) -> bool:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return r <= num_free_gpu_blocks

    # determine if last physical block is full
    # state of last physical block == last logical block before appending the previous output token
    # NOTE: this is used in def append_slot, which prepares the cache memory
    # for output in the previous iteration
    def is_last_physical_full(self, seq: Sequence) -> bool:
        num_tks_to_cache = seq.data.logical_query_len
        return (num_tks_to_cache % seq.block_size) == seq.logical_token_blocks[-1].num_tokens

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        num_logical_blocks = seq.num_new_blocks_for_inflight
        block_table = self.block_tables[seq.seq_id]

        # NOTE: logical block can be appended by an API-resume, meaning
        # the last physical block may not be full even we have new
        # logical blocks, need to check copy target
        copy_on_write_pairs = None
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        # if last block is not full but shared with otherseq
        if last_block.ref_count > 1 and not self.is_last_physical_full(seq):
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            # Decrease ref_count only
            self.gpu_allocator.free(last_block)
            copy_on_write_pairs = (last_block.block_number,
                                   new_block.block_number)
        
        # if len(block_table) < len(logical_blocks):
        #     if (self.block_sliding_window
        #             and len(block_table) >= self.block_sliding_window):
        #         # re-use a block
        #         block_table.append(block_table[len(block_table) %
        #                                        self.block_sliding_window])
        #     else:
        #         # The sequence has a new logical block.
        #         # Allocate a new physical block.
        #         block = self.gpu_allocator.allocate()
        #         block_table.append(block)
        #         return None
        if self.block_sliding_window:
            raise NotImplementedError("sliding window att not considered")
        while num_logical_blocks > len(block_table):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)

        return copy_on_write_pairs

    # NOTE: this function will only allocate for discarded blocks right before swapped blocks, not for rear new tokens
    def prepend_slot(self, seq: Sequence) -> None:
        """Allocate a physical slot for discarded tokens"""
        resumed_blocks = self.token_2_block(seq.discard_start_idx)
        num_required_allocation = self.token_2_block(seq.discard_start_idx + seq.data.running_query_len) - resumed_blocks
        block_table = self.block_tables[seq.seq_id]

        if self.block_sliding_window:
            raise NotImplementedError("sliding window att not considered")
        for i in range(num_required_allocation):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.insert(resumed_blocks + i, block)
        return

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)
    
    def get_seq_blocks_by_device(
            self, seq_group: SequenceGroup, device: Device) -> List[PhysicalTokenBlock]:
        blocks = [b for b in self._get_physical_blocks(seq_group) if b.device == device]
        return blocks

    # NOTE: consider swapped blocks and required new blocks in this function
    # NOTE: do not use this for resuming paused seqs because it assumes adding new query only
    # use required_resume_blocks instead
    def can_swap_in(self, seq_group: SequenceGroup, max_swap_blocks: int = -1, inflight_tokens: int = -1) -> bool:
        blocks = self.get_seq_blocks_by_device(seq_group, Device.CPU)
        num_required_blocks = len(blocks) if max_swap_blocks == -1 else min(len(blocks), max_swap_blocks) 
        num_required_blocks += self.required_new_blocks(seq_group, inflight_tokens)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_blocks - num_required_blocks >= self.watermark_blocks
    
    # NOTE: consider swapped blocks and discarded blocks in this function
    def required_resume_blocks(self, seq_group: SequenceGroup, max_swap_blocks: int = -1, inflight_tokens: int = -1) -> int:
        blocks = self.get_seq_blocks_by_device(seq_group, Device.CPU)
        num_required_blocks = len(blocks) if max_swap_blocks == -1 else min(len(blocks), max_swap_blocks) 
        for seq in seq_group.get_seqs(status=SequenceStatus.RESUMING):
            if inflight_tokens == -1:
                assert seq.discard_start_idx == 0
                required_allocation = self.token_2_block(seq.discard_length)
            else:
                required_allocation = self.token_2_block(seq.discard_start_idx + inflight_tokens) -\
                                      self.token_2_block(seq.discard_start_idx)
            num_required_blocks += required_allocation
        return num_required_blocks

    def swap_in(self, seq_group: SequenceGroup, max_num_blocks = -1, status=SequenceStatus.SWAPPED) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=status):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for block in block_table:
                if block.device == Device.GPU or max_num_blocks == 0:
                    new_block_table.append(block)
                    continue
                if block in mapping:
                    gpu_block = mapping[block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[block] = gpu_block
                    max_num_blocks -= 1
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup, first_n_blocks = -1) -> bool:
        blocks = self.get_seq_blocks_by_device(seq_group, Device.GPU)
        required_blocks = len(blocks)
        if first_n_blocks != -1:
            required_blocks = min(first_n_blocks, required_blocks)
        return required_blocks <= self.cpu_allocator.get_num_free_blocks()

    # swap out first n blocks, -1 means whole sequence
    def swap_out(self, seq_group: SequenceGroup, first_n_blocks = -1, status=SequenceStatus.RUNNING) -> Tuple[Dict[int, int], bool]:
        # GPU block -> CPU block.
        all_swapped_out = True
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=status):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for block in block_table:
                if block.device == Device.CPU or first_n_blocks == 0:
                    new_block_table.append(block)
                    if block.device == Device.GPU:
                        all_swapped_out = False
                    continue
                if block in mapping:
                    cpu_block = mapping[block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[block] = cpu_block
                    first_n_blocks -= 1
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping, all_swapped_out
    
    # swap out first n blocks, -1 means whole sequence
    def swap_out_from_back(
        self, 
        seq: Sequence, 
        num_swap: int,
        blocks_to_swap_out: Dict[int, int]
    ):
        # GPU block -> CPU block.
        block_table = self.block_tables[seq.seq_id]
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for i in range(len(block_table)-1, -1, -1):
            block = block_table[i]
            if block.device == Device.CPU or num_swap == 0:
                continue
            if block in mapping:
                cpu_block = mapping[block]
                cpu_block.ref_count += 1
            else:
                cpu_block = self.cpu_allocator.allocate()
                mapping[block] = cpu_block
                block_table[i] = cpu_block
                num_swap -= 1
            self.gpu_allocator.free(block)
        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        blocks_to_swap_out.update(block_number_mapping)
        return

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def _free_blocks(self, seq, num_blocks: int, block_table: BlockTable) -> int:
        cnt = 0
        remove_blocks = []
        for block in block_table:
            if block.device == Device.GPU: 
                if cnt < num_blocks:
                    self.gpu_allocator.free(block)
                    cnt += 1
                    remove_blocks.append(block)
                else: break
            else:
                if cnt < num_blocks:
                    self.cpu_allocator.free(block)
                    cnt += 1
                    remove_blocks.append(block)
                else: break
        for blk in remove_blocks:
            block_table.remove(blk) 
        if len(block_table) == 0:
            self.free(seq)
        return cnt

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def free_fst_blk_tokens(self, seq: Sequence, num_blocks) -> None:
        block_table = self.block_tables[seq.seq_id] 
        return self._free_blocks(seq, num_blocks, block_table)


    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
