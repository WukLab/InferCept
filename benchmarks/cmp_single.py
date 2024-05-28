import random
from typing import List, Optional

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from kernel import attention_ops
import argparse

NUM_SEQ = 1

MAX_Q_LEN = 64
CTX_LEN = 1024
TEST_SEED = 0


@torch.inference_mode()
def run_single_query_att(
    query,
    key_cache,
    value_cache,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    dtype,
):
    num_tokens = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    output = torch.empty(num_tokens,
                        num_heads,
                        head_size,
                        dtype=dtype,
                        device='cuda')
    # warm up
    for _ in range(10):
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            None,  # ALiBi slopes.
        )
        output, query = query, output
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    attention_ops.single_query_cached_kv_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # ALiBi slopes.
    )
    end.record()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    print(f'Latency: {ms} ms')
    
    output, query = query, output
    attention_ops.single_query_cached_kv_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # ALiBi slopes.
    )
    assert(output.size(0) == num_tokens)
    return output


def test_milti_token_single_query_kernel(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # prepare tokens
    # q_lens = random.choices(range(1, MAX_Q_LEN+1), k=num_seqs)
    q_lens = [MAX_Q_LEN] * num_seqs
    num_tokens = sum(q_lens)
    
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    query.uniform_(-1e-3, 1e-3)
    
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    num_blocks = num_seqs * CTX_LEN // block_size
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=(num_blocks, *key_block_shape),
                            dtype=dtype,
                            device='cuda')
    key_cache.uniform_(-1e-3, 1e-3)
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(size=(num_blocks, *value_block_shape),
                              dtype=dtype,
                              device='cuda')
    value_cache.uniform_(-1e-3, 1e-3)
    
    context_lens = [x + CTX_LEN - l + 1 for l in q_lens for x in range(l)]
    max_context_len = max(context_lens)
    assert(max_context_len == CTX_LEN)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')
    
    # prepare cache
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    assert(max_num_blocks_per_seq == CTX_LEN // block_size)
    # each seq will have different space
    block_tables = []
    for i, l in enumerate(q_lens):
        block_table = [[
            x + i * max_num_blocks_per_seq for x in range(max_num_blocks_per_seq)
        ]] * l
        block_tables.extend(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
    scale = float(1.0 / (head_size**0.5))
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda") 

    # last not count
    query_start_ids = [0]
    for q_len in q_lens:
        query_start_ids.append(query_start_ids[-1] + q_len)

    output = run_single_query_att(
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        dtype
      )
    
    print(output.size())

if __name__ == "__main__":
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    # for dtype in [torch.half, torch.bfloat16, torch.float]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=None)
    parser.add_argument('-q', type=int, default=None)
    args = parser.parse_args()
    NUM_SEQ = args.s or NUM_SEQ
    MAX_Q_LEN = args.q or MAX_Q_LEN
    
    dtype = torch.half
    print(f'Testing with '
        f'dtype={dtype}, '
        f'num_seq={NUM_SEQ}, '
        f'max_q_per_seq={MAX_Q_LEN}, '
        f'context_len={CTX_LEN}')
    test_milti_token_single_query_kernel(
        num_seqs=NUM_SEQ,
        num_heads=16,
        head_size=256,
        block_size=16,
        dtype=dtype   
    )