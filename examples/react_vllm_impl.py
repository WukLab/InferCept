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

class APIExecutor:
    def __init__(self) -> None:
        self._queue = queue.Queue()
    
    def _add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int):
        time.sleep(api_time)
        self._queue.put((request_id, seq_id, ret_len))
    
    def add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int):
        task = threading.Thread(target=self._add_task, args=(request_id, seq_id, api_time, ret_len))
        task.start()
        return task
    
    def _get_results(self) -> Dict[str, Dict[int, int]]:
        results = {}
        current_num_ret = self._queue.qsize()
        for _ in range(current_num_ret):
            request_id, seq_id, ret_len = self._queue.get()
            if request_id not in results:
                results[request_id] = {}
            results[request_id][seq_id] = ret_len
        return results
    
    def resume(self, vllm_engine: LLMEngine) -> None:
        api_rets = self._get_results()
        for request_id, seq_id_to_ret_len in api_rets.items():
            response = {}
            for seq_id, ret_len in seq_id_to_ret_len.items():
                response[seq_id] = [0] * ret_len
            vllm_engine.resume_request(request_id, response)
        
        

def run_vllm(
    args: argparse.Namespace,
) -> float:
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    stop = [utils.get_api_stop_string()]
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
            stop=stop,
            use_api_simulator=True,
            api_return_length=32,
            api_invoke_interval=16 + request_id,
            api_exec_time=1.0
        )
        engine.add_request(
            request_id=str(request_id),
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )

    start = time.perf_counter()
    # Run the engine.
    outputs: List[RequestOutput] = []
    iter = 0
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
            if output.paused:
                print(f'iter: {iter}, output: {output}')
                sampling_params: SamplingParams = engine.scheduler.paused[output.request_id][0].sampling_params
                for (rid, sid) in output.paused:
                    task = api_engine.add_task(output.request_id, sid, sampling_params.api_exec_time, sampling_params.api_return_length)
                    tasks.add(task)
        api_engine.resume(engine)
        iter += 1
    
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    end = time.perf_counter()
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
    print(elapsed_time)

    # total_num_tokens = sum(
    #     prompt_len + output_len for _, prompt_len, output_len in requests
    # )
    # print(
    #     f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
    #     f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    # )


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
    