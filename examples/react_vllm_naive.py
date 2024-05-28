"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple, Dict

import torch
import queue
import threading
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams, LLMEngine, EngineArgs, utils
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup, SequenceStatus

stop_token_ids = [utils.get_api_stop_token()]
request_id_2_sampling_params: Dict[str, SamplingParams] = {}

class APIExecutor:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self.prompts: Dict[str, List[int]] = {}
    
    def _add_task(self, request_id: str, api_time: float, ret_len: int, output_len: int):
        time.sleep(api_time)
        self._queue.put((request_id, ret_len, output_len))
    
    def add_task(self, request_id: str, api_time: float, ret_len: int, output_len: int, previous_token_ids: List[int]):
        self.prompts[request_id] = previous_token_ids
        task = threading.Thread(target=self._add_task, args=(request_id, api_time, ret_len, output_len))
        task.start()
        return task
    
    def resume(self, vllm_engine: LLMEngine) -> None:
        current_num_ret = self._queue.qsize()
        for _ in range(current_num_ret):
            request_id, ret_len, output_len = self._queue.get()
            previous_token_ids = self.prompts[request_id]
            ret_len = min(ret_len, vllm_engine.scheduler_config.max_model_len - len(previous_token_ids))
            ret_len = min(ret_len, output_len - 1)
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                # use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len - ret_len,
                stop_token_ids=stop_token_ids,
                use_api_simulator=True,
                api_return_length=32,
                api_invoke_interval=16 + int(request_id) % 64,
                api_exec_time=1.0
            )
            vllm_engine.add_request(
                request_id=str(int(request_id)+1024),
                prompt=None,
                sampling_params=sampling_params,
                prompt_token_ids=previous_token_ids + [0] * ret_len,
            )
            request_id_2_sampling_params[str(int(request_id)+1024)] = sampling_params

def run_vllm(
    args: argparse.Namespace,
) -> float:
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    api_engine = APIExecutor()
    tasks = set()
    
    dummy_prompt_token_ids = [[0] * args.input_len] * args.num_prompts
    
    # Add the requests to the engine.
    for request_id, prompt_token_ids in enumerate(dummy_prompt_token_ids):
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            # use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=args.output_len,
            stop_token_ids=stop_token_ids,
            use_api_simulator=True,
            api_return_length=32,
            api_invoke_interval=16 + request_id % 64,
            api_exec_time=1.0
        )
        engine.add_request(
            request_id=str(request_id),
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
        request_id_2_sampling_params[str(request_id)] = sampling_params
    pbar = tqdm(total=args.num_prompts, desc="Processed prompts")
    start = time.perf_counter()
    # Run the engine.
    outputs: List[RequestOutput] = []
    iter = 0
    while engine.has_unfinished_requests() or any([task.is_alive() for task in tasks]):
        # print(f"iter: {iter}")
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                assert len(output.outputs) == 1
                if output.outputs[0].finish_reason == "stop":
                    print(output)
                    # Make new prompt with API return
                    sampling_params: SamplingParams = request_id_2_sampling_params[output.request_id]
                    task = api_engine.add_task(
                        output.request_id, 
                        sampling_params.api_exec_time, 
                        sampling_params.api_return_length, 
                        sampling_params.max_tokens - len(output.outputs[0].token_ids),
                        output.prompt_token_ids + output.outputs[0].token_ids)
                    tasks.add(task)
                else:
                    assert output.outputs[0].finish_reason == "length"
                    outputs.append(output)
                    pbar.update(1)
        api_engine.resume(engine)
        iter += 1
    
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    end = time.perf_counter()
    pbar.close()
    for request_output in outputs:
        for seq_output in request_output.outputs:
            print(seq_output.text)
            print(seq_output.token_ids)
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    elapsed_time = run_vllm(
        args,
    )

    total_num_tokens = (args.input_len + args.output_len) * args.num_prompts
    print(
        f'time: {elapsed_time:.2f} s, '
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--input-len", type=int, default=512
    )
    parser.add_argument(
        "--output-len", type=int, default=512
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of prompts to process."
    )
    parser = EngineArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    main(args)
    