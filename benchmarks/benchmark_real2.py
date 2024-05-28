"""Benchmark offline inference throughput."""
import argparse
import csv
from datetime import datetime
import json
import random
import time
from typing import List, Optional, Tuple, Dict
import uuid

import torch
import queue
import threading
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm.rich import tqdm

from vllm import LLM, SamplingParams, LLMEngine, EngineArgs, utils
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.model_executor.layers.attention import CACHE_EVENTS
import numpy as np
import os

mu, sigma = 1, 0
log_path = './api_simu.log'

def save_to_csv(args, total_tks, total_time, total_tokens_actual, throughput, total_apis):
    with open(args.log_path, "a+", newline='') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write headers if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(['Timestamp', 'Tokens', 'Total Time(s)', 'Total Throughput', 'Experiment CSV', 'Api Policy', 'Total Apis', 'Input Length', 'Output Length', 'Num Prompts', 'API Time MIU', 'API Time SIG', 'API Return Length', 'API Max Calls', 'API Inv Offset', 'API Inv Mod', 'Model', 'Tokenizer', 'Revision', 'Tokenizer Revision', 'Tokenizer Mode', 'Trust Remote Code', 'Download Dir', 'Load Format', 'Data Type', 'Max Model Length', 'Worker Use Ray', 'Pipeline Parallel Size', 'Tensor Parallel Size', 'Block Size', 'Seed', 'Swap Space', 'GPU Memory Utilization', 'Max Num Batched Tokens', 'Max Num Seqs', 'Disable Log Stats', 'Quantization', 'Chunk Fill', 'Chunk Size', 'API Policy', 'Heuristic Coef', 'Discard Policy', 'Resize Model', 'N Layer', 'N Embed', 'N Head'])

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write data to CSV
        csvwriter.writerow([
            timestamp,
            total_tks,
            total_time,
            throughput,

            args.exp_json,
            args.api_policy,
            total_apis,

            args.input_len,
            args.output_len,
            args.num_prompts,
            args.api_time_miu,
            args.api_time_sig,
            args.api_ret_len,
            args.api_max_calls,
            args.api_inv_offset,
            args.api_inv_mod,
            args.model,
            args.tokenizer,
            args.revision,
            args.tokenizer_revision,
            args.tokenizer_mode,
            args.trust_remote_code,
            args.download_dir,
            args.load_format,
            args.dtype,
            args.max_model_len,
            args.worker_use_ray,
            args.pipeline_parallel_size,
            args.tensor_parallel_size,

            args.seed,
            args.swap_space,
            args.gpu_memory_utilization,
            args.max_num_batched_tokens,
            args.max_num_seqs,
            args.disable_log_stats,
            args.quantization,
            args.chunk_fill,
            args.chunk_size,
            args.heuristic_coef,
            args.discard_policy,
            args.resize_model,
            args.n_layer,
            args.n_embed,
            args.n_head,
        ])

class APIExecutor:
    def __init__(self) -> None:
        self._queue = []
        self._waiting = []
        self.this_iter_waiting = []
        self._total_apis = 0
        self.exp_status = {}
        self.exp_json = {}
        self.stop = None
        self.args = None
        self.curr_time = None
        self.req_stop = {

        }

    
    def _add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int, stop_time: float):
        # time.sleep(max(0, api_time))
        self.req_stop[request_id] = (stop_time, api_time)
        self.this_iter_waiting.append((request_id, seq_id, ret_len))
    
    def add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int, stop_time: float):
        # task = threading.Thread(target=self._add_task, args=(request_id, seq_id, api_time, ret_len))
        # task.start()
        self._add_task(request_id, seq_id, api_time, ret_len, stop_time)
        self._total_apis += 1
        #return task
    
    def _get_results(self) -> Dict[str, Dict[int, int]]:
        results = {}
        waiting_queue = []
        resume_queue = []
        for (request_id, seq_id, ret_len) in self._waiting:
            stop_time, api_time = self.req_stop[request_id]
            if stop_time + api_time < self.curr_time:
                resume_queue.append((request_id, seq_id, ret_len))
            else:
                waiting_queue.append((request_id, seq_id, ret_len))
        for item in resume_queue:
            request_id, seq_id, ret_len = item
            if request_id not in results:
                results[request_id] = {}
            results[request_id][seq_id] = ret_len
        self._waiting = waiting_queue
        return results

    def get_new_sampling_params(self, request_id):
        self.exp_status[request_id] += 1
        new_sampling_params, _ = get_sampling_param(self.args, self.exp_json[request_id], self.exp_status[request_id], self.stop)
        return new_sampling_params

    def resume(self, vllm_engine: LLMEngine, stop_time) -> None:
        self.curr_time = stop_time
        api_rets = self._get_results()
        for request_id, seq_id_to_ret_len in api_rets.items():
            response = {}
            for seq_id, ret_len in seq_id_to_ret_len.items():
                response[seq_id] = [0] * ret_len
            sampling_param = self.get_new_sampling_params(request_id)
            vllm_engine.resume_request(request_id, response, sampling_param)
        self._waiting.extend(self.this_iter_waiting)
        self.this_iter_waiting = []

def get_sampling_param(args, experiments, experiment_num, stop, initial=False):
    # Format of each experiment
    if experiment_num >= len(experiments):
        experiment = {"prompt_size": 0, "api_exec_time": 0}
    else:
        experiment = experiments[experiment_num]
    prompt_size, api_exec_time, api_return_len = experiment["prompt_size"], experiment.get("api_exec_time", 0), experiment.get("api_return_len", 0)
    api_invoke_interval = prompt_size
    if api_exec_time == 0 and api_return_len == 0:
        api_max_calls = 0
    else:
        api_max_calls = 1
    
    if api_invoke_interval + api_return_len >= 2048:
        api_return_len = 0
        api_invoke_interval = min(api_invoke_interval, 2048)
    
    if initial:
        api_invoke_interval = 1
    
    return  SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        # use_beam_search=use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
        stop=stop,
        use_api_simulator=True,
        
        api_return_length=api_return_len,
        api_invoke_interval=api_invoke_interval,

        api_exec_time=api_exec_time,
        api_max_calls=api_max_calls
    ), prompt_size

def run_vllm(
    args: argparse.Namespace,
) -> float:
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    stop = [utils.get_api_stop_string()]

    api_engine = APIExecutor()
    api_engine.stop = stop
    api_engine.args = args

    tasks = set()
    with open(args.exp_json) as f:
        exp_json = json.load(f)
    # request_id -> prop    
    # Fill the number of total request_ids == args.num_prompts. Using random sampling
    # to sample from the experiments.
    keys_to_sample = random.choices(list(exp_json.keys()), k=args.num_prompts)
    new_exp_json = {}
    i = 0
    for key in keys_to_sample:
        new_unique_request_id =  str(i)
        i += 1
        new_exp_json[new_unique_request_id] = exp_json[key].copy()
    
    exp_json = new_exp_json
    current_num_prompts = len(exp_json.keys())
    # Tokenize prompts/api outputs if provided as strings
    total_tokens_actual = 0
    for request_id, experiments in exp_json.items():
        for experiment in experiments:
            if "prompt" in experiment and "prompt_size" not in experiment:
                experiment["prompt_size"] = len(engine.tokenizer.encode(experiment["prompt"]))
            if "api_output" in experiment and "api_return_len" not in experiment:
                experiment["api_return_len"] = len(engine.tokenizer.encode(experiment["api_output"]))
            total_tokens_actual += experiment.get("prompt_size", 0) + experiment.get("api_return_len", 0)

    # TODO Remove
    with open(args.exp_json + "_filtered.json", "w") as f:
        json.dump(exp_json, f, indent=4)

    print("Running a total of {} prompts {}".format(current_num_prompts, total_tokens_actual))
    exp_status = {}
    for request_id, experiments in exp_json.items():
        exp_status[request_id] = 0
        sampling_params, prompt_size = get_sampling_param(args, experiments, exp_status[request_id], stop, initial=True)

        prompt_token_ids = [0] * prompt_size
        engine.add_request(
            request_id=str(request_id),
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
    api_engine.exp_status = exp_status
    api_engine.exp_json = exp_json

    pbar = tqdm(total=args.num_prompts, desc="Processed prompts")
    start = time.perf_counter()
    # Run the engines.
    outputs: List[RequestOutput] = []
    iter = 0
    torch.cuda.cudart().cudaProfilerStart()
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        stop_time = time.monotonic()
        can_stop = False
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)
                if len(outputs) == 100:
                    start = time.perf_counter()
                if len(outputs) == 400:
                    can_stop = True
            if output.paused:
                # print(f'iter: {iter}, output: {output}')
                sampling_params: SamplingParams = engine.scheduler.paused[output.request_id][0].sampling_params
                for (rid, sid) in output.paused:
                    # task = api_engine.add_task(output.request_id, sid, sampling_params.api_exec_time, sampling_params.api_return_length)
                    # tasks.add(task)
                    api_engine.add_task(output.request_id, sid, sampling_params.api_exec_time, sampling_params.api_return_length, stop_time)
        api_engine.resume(engine, stop_time)
        iter += 1
        if can_stop:
            break
    
    torch.cuda.cudart().cudaProfilerStop()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    # outputs = sorted(outputs, key=lambda x: int(x.request_id))
    end = time.perf_counter()
    pbar.close()
    print(f"total apis: {api_engine._total_apis}")
    # for request_output in outputs:
    #     for seq_output in request_output.outputs:
    #         print(seq_output.text)
    #         print(seq_output.token_ids)
    print(f'total tokens: {engine.scheduler.total_tks}')
    with open('history.log', 'w+') as f:
        for h in engine.scheduler.history:
            f.write(str(h) + '\n')
    return end - start, engine.scheduler.total_tks, total_tokens_actual, api_engine._total_apis

def main(args: argparse.Namespace):
    print(args, "Seed", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    elapsed_time, total_tks, total_tokens_actual, total_apis = run_vllm(
        args,
    )
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in CACHE_EVENTS])
    print(f'cache events time, mean: {times.mean():<10} max: {times.max():<10} min: {times.min():<10} total:  {times.sum():<10}')
    total_num_tokens = total_tokens_actual
    save_to_csv(args, total_tks, elapsed_time, total_tokens_actual, total_num_tokens / elapsed_time, total_apis)
    print(
        f"###### RUN ########"
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
        "--num-prompts", type=int, default=100, help="Number of prompts to process."
    )
    parser.add_argument(
        "--api-time-miu", type=float, default=6, help="API execution time mean."
    )
    parser.add_argument(
        "--api-time-sig", type=float, default=2, help="API execution time sigma."
    )
    parser.add_argument(
        "--api-ret-len", type=int, default=128, help="API return length."
    )
    parser.add_argument(
        "--api-max-calls", type=int, default=4, help="API max calls. -1 means no limit."
    )
    parser.add_argument(
        "--api-inv-offset", type=int, default=64, help="API invocation offset."
    )
    parser.add_argument(
        "--api-inv-mod", type=int, default=32, help="API invocation offset."
    )
    parser.add_argument(
        "--log", type=str, default=None,
    )
    parser.add_argument(
        "--exp-json", type=str, default="exp_version2/basic.json"
    )
    parser = EngineArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    mu, sigma = args.api_time_miu, args.api_time_sig
    if args.log:
        args.log_path = args.log
    main(args)
    
# total1: 46039
