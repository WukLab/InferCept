"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple, Dict

import csv
import torch
import queue
import threading
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams, LLMEngine, EngineArgs, utils
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup, SequenceStatus

import numpy as np

mu, sigma = 1, 0

stop_token_ids = [utils.get_api_stop_token()]
request_id_2_sampling_params: Dict[str, SamplingParams] = {}


configs_list_indices = [1, 12, 17, 12, 0, 11, 5, 7, 2, 9, 2, 23, 2, 3, 4, 21, 4, 6, 20, 12, 2, 2, 21, 6, 4, 0, 12, 19, 7, 15, 13, 16, 20, 12, 18, 21, 7, 3, 21, 19, 20, 23, 19, 12, 17, 13, 9, 4, 2, 12, 16, 12, 11, 19, 16, 17, 15, 11, 6, 21, 23, 2, 3, 11, 7, 10, 6, 8, 18, 19, 6, 14, 2, 3, 14, 11, 0, 13, 22, 7, 1, 11, 21, 18, 2, 14, 22, 18, 13, 9, 6, 5, 11, 16, 23, 18, 17, 13, 9, 2, 14, 1, 16, 6, 15, 11, 13, 2, 10, 15, 18, 4, 19, 6, 19, 16, 4, 4, 22, 9, 13, 17, 23, 18, 7, 1, 3, 6, 3, 23, 17, 14, 1, 13, 23, 9, 14, 12, 20, 14, 8, 16, 7, 21, 10, 21, 22, 16, 8, 16, 19, 4, 1, 2, 8, 7, 9, 4, 9, 18, 7, 10, 10, 16, 8, 0, 6, 2, 18, 16, 16, 3, 8, 22, 5, 23, 1, 20, 5, 21, 10, 19, 22, 12, 16, 3, 6, 2, 16, 15, 10, 8, 0, 21, 5, 2, 13, 6, 2, 13, 17, 6, 12, 20, 0, 8, 20, 9, 17, 9, 5, 14, 8, 23, 10, 17, 19, 15, 1, 0, 8, 13, 9, 23, 20, 1, 5, 23, 4, 4, 16, 16, 15, 0, 4, 3, 11, 2, 11, 9, 15, 18, 8, 6, 0, 7, 20, 7, 13, 9, 17, 12, 4, 21, 10, 12, 1, 8, 17, 21, 9, 11, 0, 2, 22, 10, 6, 21, 22, 6, 18, 7, 20, 5, 22, 8, 16, 18, 5, 2, 23, 9, 4, 21, 14, 21, 4, 11, 11, 8, 23, 11, 11, 18, 18, 8, 15, 10, 2, 3, 9, 6, 6, 0, 2, 22, 17, 21, 12, 1, 4, 23, 14, 11, 20, 14, 7, 17, 16, 21, 5, 11, 21, 9, 22, 2, 14, 22, 10, 5, 6, 2, 11, 20, 5, 23, 2, 8, 4, 8, 12, 14, 22, 10, 11, 13, 8, 4, 20, 22, 13, 19, 13, 13, 19, 9, 1, 21, 12, 7, 18, 13, 16, 13, 17, 17, 20, 19, 17, 6, 14, 15, 13, 0, 6, 1, 18, 22, 9, 13, 22, 10, 23, 16, 2, 0, 7, 0, 20, 3, 18, 5, 2, 12, 16, 16, 8, 23, 2, 11, 14, 10, 21, 5, 23, 16, 17, 16, 7, 18, 4, 5, 22, 9, 17, 10, 19, 18, 0, 22, 4, 22, 1, 10, 14, 7, 21, 13, 13, 12, 5, 10, 1, 22, 2, 16, 1, 19, 13, 16, 22, 21, 18, 14, 3, 6, 10, 9, 9, 5, 0, 10, 14, 3, 6, 3, 3, 12, 19, 23, 17, 20, 5, 6, 19, 14, 0, 2, 16, 9, 19, 3, 19, 7, 20, 22, 14, 4, 14, 3, 17, 12, 5, 22, 8, 7, 11, 16, 2, 7, 0, 18, 6, 15, 21, 5, 6, 23, 8, 18]

class APIExecutor:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self.prompts: Dict[str, List[int]] = {}
        self._total_apis = 0
    
    def _add_task(self, request_id: str, api_time: float, ret_len: int, output_len: int, api_max_calls: int):
        time.sleep(api_time)
        self._queue.put((request_id, ret_len, output_len, api_time, api_max_calls))
    
    def add_task(self, request_id: str, api_time: float, ret_len: int, output_len: int, api_max_calls: int, previous_token_ids: List[int]):
        self.prompts[request_id] = previous_token_ids
        task = threading.Thread(target=self._add_task, args=(request_id, api_time, ret_len, output_len, api_max_calls))
        task.start()
        self._total_apis += 1
        return task
    
    def resume(self, vllm_engine: LLMEngine) -> None:
        current_num_ret = self._queue.qsize()
        for _ in range(current_num_ret):
            request_id, ret_len, output_len, api_time, api_max_calls = self._queue.get()
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
                api_return_length=ret_len,
                api_invoke_interval=args.api_inv_offset + int(request_id) % args.api_inv_mod,
                api_exec_time=api_time,
                api_max_calls=max(0, api_max_calls-1)
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

    # configs
    config_list = []
    if args.mixed:
        configs = csv.DictReader(open("configs.csv"))
        for row in configs:
            new_row = {} 
            for k,v in row.items():
                new_row[k] = int(v)
            config_list.append(new_row)
        
        # generate indices
        indices = configs_list_indices
        with open("run2.log", "a+") as logfile: 
            logfile.write(f'\n{indices}')
    
    # Add the requests to the engine.
    for request_id, prompt_token_ids in enumerate(dummy_prompt_token_ids):
        if args.mixed:
            config = config_list[indices[request_id]]
        api_exec_time = config['api_exec_time'] if args.mixed else abs(np.random.normal(mu, sigma))
        api_max_calls = config['api_max_calls'] if args.mixed else args.api_max_calls
        api_ret_len = config['api_ret_len'] if args.mixed else args.api_ret_len
        api_inv_offset = config['api_inv_offset'] if args.mixed else args.api_inv_offset
        api_inv_mod = config['api_inv_mod'] if args.mixed else args.api_inv_mod
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            # use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=args.output_len,
            stop_token_ids=stop_token_ids,
            use_api_simulator=True,
            api_return_length=api_ret_len,
            api_invoke_interval=api_inv_offset + request_id % api_inv_mod,
            api_exec_time=api_exec_time,
            api_max_calls=api_max_calls
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
    torch.cuda.cudart().cudaProfilerStart()
    while engine.has_unfinished_requests() or any([task.is_alive() for task in tasks]):
        # print(f"iter: {iter}")
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                assert len(output.outputs) == 1
                if output.outputs[0].finish_reason == "stop" and output.outputs[0].token_ids[-1] in stop_token_ids:
                    # print(output)
                    # Make new prompt with API return
                    sampling_params: SamplingParams = request_id_2_sampling_params[output.request_id]
                    task = api_engine.add_task(
                        output.request_id, 
                        sampling_params.api_exec_time, 
                        sampling_params.api_return_length, 
                        sampling_params.max_tokens - len(output.outputs[0].token_ids),
                        sampling_params.api_max_calls, 
                        output.prompt_token_ids + output.outputs[0].token_ids)
                    tasks.add(task)
                else:
                    assert output.outputs[0].finish_reason == "length"
                    outputs.append(output)
                    pbar.update(1)
        api_engine.resume(engine)
        iter += 1
    
    torch.cuda.cudart().cudaProfilerStop()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    end = time.perf_counter()
    pbar.close()
    print(f"total apis: {api_engine._total_apis}")
    print(f'total tokens: {engine.scheduler.total_tks}')
    # for request_output in outputs:
    #     for seq_output in request_output.outputs:
    #         print(seq_output.text)
    #         print(seq_output.token_ids)
    return end - start, engine.scheduler.total_tks


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    elapsed_time, total_tks = run_vllm(
        args,
    )

    total_num_tokens = (args.input_len + args.output_len) * args.num_prompts
    with open("run2.log", "a") as logfile: 
        logfile.write(
            f"\n###### RUN NAIVE ########\n"
            f'tokens; {total_tks}\n'
            f'args: {args}\n'
            f'time: {elapsed_time:.2f} s, \n'
            f"{total_num_tokens / elapsed_time:.2f} tokens/s\n"
            f"###### RUN ########"
        )
    print(
        f"###### RUN NAIVE ########"
        f'args: {args}'
        f'time: {elapsed_time:.2f} s, '
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
        f"###### RUN ########"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--input-len", type=int, default=1024
    )
    parser.add_argument(
        "--output-len", type=int, default=1024
    )
    parser.add_argument(
        "--num-prompts", type=int, default=500, help="Number of prompts to process."
    )
    parser.add_argument(
        "--api-time-miu", type=float, default=0.1, help="API execution time mean."
    )
    parser.add_argument(
        "--api-time-sig", type=float, default=0.0, help="API execution time sigma."
    )
    parser.add_argument(
        "--api-ret-len", type=int, default=128, help="API return length."
    )
    parser.add_argument(
        "--api-max-calls", type=int, default=4, help="API max calls."
    )
    parser.add_argument(
        "--api-inv-offset", type=int, default=128, help="API invocation offset."
    )
    parser.add_argument(
        "--api-inv-mod", type=int, default=1, help="API invocation offset."
    )
    parser.add_argument(
        "--mixed", action='store_true', help="Mixed batch"
    )
    parser = EngineArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    mu, sigma = args.api_time_miu, args.api_time_sig
    main(args)
    
