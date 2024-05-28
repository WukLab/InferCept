"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

from vllm import LLM, SamplingParams, LLMEngine, EngineArgs, utils
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput



def run_vllm(
    args: argparse.Namespace,
) -> float:
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    
    dummy_prompt_token_ids = [[0] * args.input_len] * args.num_prompts

    # Add the requests to the engine.
    for request_id, prompt_token_ids in enumerate(dummy_prompt_token_ids):
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=args.output_len,
            use_api_simulator=True
        )
        engine.add_request(
            request_id=str(request_id),
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )

    start = time.perf_counter()
    # FIXME(woosuk): Do use internal method.
    num_requests = engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")
    # Run the engine.
    outputs: List[RequestOutput] = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)
    pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    end = time.perf_counter()
    print(f'total tokens: {engine.scheduler.total_tks}')
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.

    elapsed_time = run_vllm(args)
    total_num_tokens = (args.input_len + args.output_len) * args.num_prompts
    print(
        f'time: {elapsed_time:.2f} s, '
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument(
        "--input-len", type=int, default=1024
    )
    parser.add_argument(
        "--output-len", type=int, default=1
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    main(args)
