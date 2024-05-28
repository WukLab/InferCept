"""Benchmark offline inference throughput."""
import argparse
import random
import time
from typing import List, Dict, Tuple

import torch
import queue
import threading
from tqdm.rich import tqdm
from vllm import SamplingParams, LLMEngine, EngineArgs, utils
from vllm.outputs import RequestOutput
from vllm.model_executor.layers.attention import CACHE_EVENTS
import numpy as np
import csv
from datetime import datetime
import json
from pathlib import Path
import os

class Request:
    def __init__(self, request_id, prompt, sampling_params, prompt_token_ids):
        self.request_id = request_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.prompt_token_ids = prompt_token_ids
        self.start_time = -1
        self.start_generate = -1
        self.end_time = -1
        self.start_length = 0
        self.end_length = 0
        self.pause_time = -1
        self.resume_time = -1
        self.finished = False
        self.api_times = []

class APIExecutor:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self._total_apis = 0
        self.exp_json = {}
        self.exp_status = {}
        self.total_output_toks = {}
        self.args = None
        self.stop = None
    
    def _add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int):
        time.sleep(api_time)
        self._queue.put((request_id, seq_id, ret_len))
    
    def add_task(self, request_id: str, seq_id: int, api_time: float, ret_len: int):
        task = threading.Thread(target=self._add_task, args=(request_id, seq_id, api_time, ret_len))
        task.start()
        self._total_apis += 1
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
    
    def get_new_sampling_params(self, request_id: str) -> Tuple[SamplingParams, int]:
        # if self.args.no_api:
        #     new_sampling_params = SamplingParams(
        #         n=1,
        #         temperature=0.0,
        #         top_p=1.0,
        #         # use_beam_search=use_beam_search,
        #         ignore_eos=True,
        #         max_tokens=min(2048, self.exp_json[request_id]['output_len']),
        #     )
        #     return new_sampling_params, self.exp_json[request_id]['prompt_size']
        if request_id not in self.exp_status:
            self.exp_status[request_id] = 0
        else:
            self.exp_status[request_id] += 1
        
        experiments = self.exp_json[request_id]
        experiment_num = self.exp_status[request_id]
        # Format of each experiment
        if experiment_num >= len(experiments):
            experiment = {"prompt_tokens": 0, "completion_tokens": 0, "api_time": 0, 'api_token_length': 0}
        else:
            experiment = experiments[experiment_num]
        if "api_time" not in experiment:
            assert "api_token_length" not in experiment
            experiment["api_time"] = 0
            experiment["api_token_length"] = 0
        prompt_size, completion_tokens, api_exec_time, api_return_len = experiment["prompt_tokens"], experiment["completion_tokens"], experiment["api_time"], experiment["api_token_length"]
        api_invoke_interval = completion_tokens
        
        if api_exec_time == 0 and api_return_len == 0:
            api_max_calls = 0
        else:
            api_max_calls = 1
        
        # if api_invoke_interval + api_return_len >= 2048:
        #     api_return_len = 0
        #     api_invoke_interval = min(api_invoke_interval, 2048)
        
        # if experiment_num == 0:
        #     api_invoke_interval = 1
        new_sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            # use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=min(2048, self.total_output_toks[request_id]),
            
            stop=self.stop,
            use_api_simulator=True,
            
            api_return_length=api_return_len,
            api_invoke_interval=api_invoke_interval,

            api_exec_time=api_exec_time,
            api_max_calls=api_max_calls
        )
        return new_sampling_params, prompt_size
    
    def resume(self, vllm_engine: LLMEngine, requests, start_measure) -> None:
        api_rets = self._get_results()
        resume_time = time.perf_counter()
        for request_id, seq_id_to_ret_len in api_rets.items():
            response = {}
            if start_measure:
                # num_apis[request_id] = (num_apis[request_id][0]+1, num_apis[request_id][1])
                # latest_api_resume[request_id] = time.perf_counter()
                # pause_time = num_apis[request_id][-1][0]
                # num_apis[request_id][-1] = (pause_time, resume_time)
                r: Request = requests[int(request_id)]
                r.resume_time = resume_time
                r.api_times.append((r.pause_time, r.resume_time))
                r.pause_time = -1
                r.resume_time = -1
            for seq_id, ret_len in seq_id_to_ret_len.items():
                response[seq_id] = [0] * ret_len
            sampling_params, _ = self.get_new_sampling_params(request_id)
            vllm_engine.resume_request(request_id, response, sampling_params)
    
    def generate_exec_times(self, distro, num_prompts, seed):
        rng = np.random.default_rng(seed)
        if distro == 'N':
            # normal distribution
            return np.abs(rng.normal(loc=11, scale=3, size=(num_prompts,)))
        elif distro == 'U':
            # uniform distribution
            return rng.uniform(low=0.1, high=20, size=(num_prompts,))
        else:
            # Generate random numbers from gamma distribution
            right = np.abs(rng.gamma(shape=0.5, scale=4, size=(num_prompts,)))  # shorter api times
            left = np.abs(20-right)                                             # longer api times
            if distro == 'L':
                return left
            elif distro == 'R':
                return right
            elif distro == 'B':
                return np.concatenate([rng.choice(left, num_prompts//2),
                                       rng.choice(right, num_prompts//2)])
            else:
                return ValueError(f'Unsupported distribution: {distro}')
            


def run_vllm(
    args: argparse.Namespace,
) -> float:
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    if args.no_api:
        stop = []
    else:
        stop = utils.get_api_stop_strings()
    api_engine = APIExecutor()
    api_engine.args = args
    api_engine.stop = stop
    tasks = set()
    
    args.num_prompts = 4 * int(args.qps * args.window)
    # dummy_prompt_token_ids = [[0] * args.input_len] * args.num_prompts
    # api_exec_times = api_engine.generate_exec_times(args.distro, args.num_prompts, args.seed)
    
    # prepare input tokens
    with open(args.exp_json) as f :
        exp_json = json.load(f)
    # request_id -> prop    
    # Fill the number of total request_ids == args.num_prompts. Using random sampling
    # to sample from the experiments.
    # Use continuous int as request id now for easy tracking
    if args.no_api:
        keys_to_sample = list(exp_json.keys())
        assert len(keys_to_sample) == args.num_prompts
    else:
        keys_to_sample = random.choices(list(exp_json.keys()), k=args.num_prompts)
    new_exp_json = {}
    new_2_old_key = {}
    for i, key in enumerate(keys_to_sample):
        new_exp_json[str(i)] = exp_json[key].copy()
        new_2_old_key[i] = key
        
    exp_json = new_exp_json
    api_engine.exp_json = exp_json
    
    prompt_lens = []
    output_lens = []
    num_api_calls = []
    ret_lens = []
    api_exec_times = []

    # Unify chatbot with other workload
    for request_id, experiments in exp_json.items():
        ts = 0
        num_calls = 0
        for i, experiment in enumerate(experiments):    
            if "prompt" in experiment and "prompt_tokens" not in experiment:
                experiment["prompt_tokens"] = len(engine.tokenizer.encode(experiment["prompt"]))
                if i == 0 and experiment['prompt_tokens'] >= 2048:
                    print(f'long prompt: {new_2_old_key[int(request_id)]}')
            if i == 0:
                prompt_lens.append(experiment['prompt_tokens'])

            if "completion" in experiment and "completion_tokens" not in experiment:
                experiment["completion_tokens"] = len(engine.tokenizer.encode(experiment["completion"]))
                if experiment['completion_tokens'] >= 2048:
                    print(f'long completion: {new_2_old_key[int(request_id)]}')

            if "api_token" in experiment and "api_token_length" not in experiment:
                experiment["api_token_length"] = len(engine.tokenizer.encode(experiment["api_token"]))
                if experiment["api_token_length"] >= 2048:
                    print(f'long ret: {new_2_old_key[int(request_id)]}')
            
            ts += experiment["completion_tokens"]
            if "api_token_length" in experiment:
                ts += experiment["api_token_length"]
                ret_lens.append(experiment["api_token_length"])
                num_calls += 1
            
            if "api_time" in experiment:
                api_exec_times.append(experiment['api_time'])
            api_engine.total_output_toks[request_id] = ts
        output_lens.append(min(ts, 2048 - prompt_lens[-1]))
        num_api_calls.append(num_calls)
    
    with open('data_dist.csv', 'w+') as f:
        for i in range(len(prompt_lens)):
            f.write(f'{prompt_lens[i]},{output_lens[i]},{num_api_calls[i]},{ret_lens[i]},{api_exec_times[i]}\n')
        
        for j in range(i, len(ret_lens)):
            f.write(f'0,0,0,{ret_lens[j]},{api_exec_times[j]}\n')
    
    requests: List[Request] = []
    rng = np.random.default_rng(args.seed)
    arrival_times = []
    start_offset = 0
    #num_apis = {}

    # Add the requests to the engine.
    for request_id, experiments in exp_json.items():
        sampling_params, prompt_size = api_engine.get_new_sampling_params(request_id)
        prompt_token_ids = [0] * prompt_size
        requests.append(Request(
            request_id=request_id,
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids
        ))
        offset = rng.exponential(1.0 / args.qps)
        start_offset += offset
        arrival_times.append(start_offset)

    pbar = tqdm(total=args.num_prompts, desc="Processed prompts")
    start = time.perf_counter()

    index = 0

    for r in requests[:int(args.qps)+1]:
        engine.add_request(r.request_id, r.prompt, r.sampling_params, r.prompt_token_ids)
        arrival_times.pop(0)
    index = int(args.qps)+1

    # track latency
    # start_times = {}
    # end_times = {}
    # latest_api_resume = {}

    # Run the engine.
    outputs: List[RequestOutput] = []
    iter = 0
    torch.cuda.cudart().cudaProfilerStart()
    tokens_after_start = 0
    tokens_after_stop = 0
    start_measure_time = time.perf_counter()
    num_reqs = 0
    started_measure = True
    cpu_full = False
    while engine.has_unfinished_requests() or arrival_times:
        if engine.cpu_full:
            cpu_full = True
        curr_time = time.perf_counter()
        while arrival_times and start+arrival_times[0] <= curr_time:
            r = requests[index]
            engine.add_request(r.request_id, r.prompt, r.sampling_params, r.prompt_token_ids)
            arrival_times.pop(0)
            if started_measure:
                r.start_time = curr_time
                r.start_length = 0
            index += 1
        
        if not engine.has_unfinished_requests:
            api_engine.resume(engine)
            continue

        step_outputs = engine.step()
        can_stop = False
        if not started_measure and curr_time-start > 30:
            # tokens_after_start = engine.scheduler.get_tokens_have_seen() + sum(len(output.prompt_token_ids) + len(output.outputs[0].token_ids) for output in outputs)
            # start_measure_time = curr_time
            # num_reqs = len(outputs)
            started_measure = True
        for output in step_outputs:
            r: Request = requests[int(output.request_id)]
            if r.start_generate == -1:
                r.start_generate = curr_time
            if r.start_time != -1:
                r.end_time = curr_time
                r.end_length = len(output.outputs[0].token_ids)
            else:
                if started_measure:
                    r.start_time = curr_time
                    r.start_length = len(output.outputs[0].token_ids)
            if output.finished:
                outputs.append(output)
                r.finished = True
                    # if len(end_times) == 100:
                    #     can_stop = True
                pbar.update(1)
            if output.paused:
                # print(f'iter: {iter}, output: {output}')
                sampling_params: SamplingParams = engine.scheduler.paused[output.request_id][0].sampling_params

                r.pause_time = curr_time
                
                for (rid, sid) in output.paused:
                    task = api_engine.add_task(output.request_id, sid, sampling_params.api_exec_time, sampling_params.api_return_length)
                    tasks.add(task)
        api_engine.resume(engine, requests, started_measure)
        iter += 1
        if curr_time-start > args.window:
            can_stop = True
            # print(tokens_after_stop)
        if can_stop:
            break
    tokens_after_stop = engine.scheduler.get_tokens_have_seen() + sum(len(output.prompt_token_ids) + len(output.outputs[0].token_ids) for output in outputs)
    num_reqs = len(outputs)
    torch.cuda.cudart().cudaProfilerStop()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    end = time.perf_counter()
    pbar.close()
    actual_tokens = tokens_after_stop - tokens_after_start

    # parse latencies
    latencies = []
    queueing_latency = []
    tpots = []
    with open(f'./{args.msg}/api_trace.csv', 'w+') as t:
        for r in requests:
            t.write(f'{r.request_id}: {r.start_time}, {r.end_time}\n -- {r.api_times}\n')
    with open(f'./{args.msg}/all_trace.csv', 'w+') as t:
        for r in requests:
            # if r.end_length != args.output_len + 1:
            if r.start_time == -1:
                continue
            if r.end_time == -1:
                # final_norm_latency = (end - r.start_time) / 1e-5
                # latencies.append(final_norm_latency)
                continue
            if r.start_length == r.end_length:
                continue

            if not r.finished:
                r.end_time = end

            end_time, end_length = r.end_time, r.end_length
            start_time, start_length = r.start_time, r.start_length
            start_generate = r.start_generate

            api_time = 0
            apis_list = r.api_times
            while apis_list:
                api_start, api_end = apis_list.pop(0)
                bumped_start = max(api_start, start_time)
                bumped_end = min(api_end, end_time)
                api_time += bumped_end - bumped_start

            if start_generate != -1:
                queueing = start_generate - start_time
                tpot = (end_time - start_generate - api_time) / end_length
                queueing_latency.append(queueing)
                tpots.append(tpot)
            else:
                queueing = end - start_time
            
            final_norm_latency = (end_time-start_time-api_time)/(end_length-start_length)
            # final_norm_latency = (end_time-start_time)/(end_length-start_length)
            latencies.append(final_norm_latency)
            t.write(f'{r.request_id}, {r.finished}, {r.end_time-r.start_time}, {api_time}, {queueing}, {start_length}, {end_length}, {final_norm_latency}\n')

    p50_queue = np.percentile(queueing_latency, 50)
    p90_queue = np.percentile(queueing_latency, 90)
    p99_queue = np.percentile(queueing_latency, 99)
    p50_tpot = np.percentile(tpots, 50)
    p90_tpot = np.percentile(tpots, 90)
    p99_tpot = np.percentile(tpots, 99)
    
    p50_lat = np.percentile(latencies, 50)
    p90_lat = np.percentile(latencies, 90)
    p99_lat = np.percentile(latencies, 99)

    metrics = [p50_queue, p90_queue, p99_queue, p50_tpot, p90_tpot, p99_tpot, p50_lat, p90_lat, p99_lat]
    
    with open(f'./{args.msg}/iter_history.log', 'w+') as w:
        # assert len(engine.iter_times) == len(engine.scheduler.iter_history)
        print(f'engine iters: {len(engine.iter_times)}, scheduler iters: {len(engine.scheduler.iter_history)}')
        for (i_s, i_e), b in zip(engine.iter_times, engine.scheduler.iter_history):
            if i_s is not None:
                b.iter_time = i_s.elapsed_time(i_e)
            w.write(f'{b}\n')

    return end - start_measure_time, engine.scheduler.total_tks, actual_tokens, cpu_full, num_reqs, np.mean(latencies), metrics


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    elapsed_time, total_tks, actual_tks, cpu_full, num_reqs, mean_normalized_lat, metrics = run_vllm(
        args,
    )
    
    torch.cuda.synchronize()
    with open(f'./{args.msg}/results.log', "a+") as logfile: 
        logfile.write(
            f"{args.api_policy},{args.distro},{args.qps},{mean_normalized_lat:.4f}\n"
            f"\t{metrics}\n"
        )
    with open(f'./{args.msg}/dump.log', "a+") as logfile: 
        logfile.write(
            f"###### RUN ########\n"
            f'actual tokens: {actual_tks}\n'
            f'total tokens: {total_tks}\n'
            f'distribution: {args.distro}\n'
            f'args: {args}\n'
            f'time: {elapsed_time:.2f} s\n'
            f"{actual_tks / elapsed_time:.2f} tokens/s\n"
            f"finished: {num_reqs}\n"
            f"Was CPU over 98% utilized: {cpu_full}\n"
        )

    if CACHE_EVENTS:
        times = []
        with open(f'./{args.msg}/swap_wait.csv', 'w+') as f:
            for s, e, t, i, o in CACHE_EVENTS:
                times.append(s.elapsed_time(e))
                f.write(f'{times[-1]}, {t}, {i}, {o}\n')
        print(f'cache events time mean: {np.mean(times):<10} max: {np.max(times):<10} min: {np.min(times):<10} total:  {np.sum(times):<10}')
        with open(f'./{args.msg}/dump.log', "a+") as logfile: 
            logfile.write(
                f'cache events time mean: {np.mean(times):<10} max: {np.max(times):<10} min: {np.min(times):<10} total:  {np.sum(times):<10}\n'
                f"###### RUN ########\n"
        )
        
    print(f"{args.api_policy},{args.distro},{args.qps},{num_reqs / elapsed_time:.4f} q/s, \n{metrics}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--input-len", type=int, default=1024
    )
    parser.add_argument(
        "--output-len", type=int, default=1024
    )
    parser.add_argument(
        "--num-prompts", type=int, default=200000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--api-ret-len", type=int, default=48, help="API return length."
    )
    parser.add_argument(
        "--api-max-calls", type=int, default=16, help="API max calls. -1 means no limit."
    )
    parser.add_argument(
        "--api-inv-offset", type=int, default=16, help="API invocation offset."
    )
    parser.add_argument(
        "--api-inv-mod", type=int, default=1, help="API invocation offset."
    )
    parser.add_argument(
        "--qps", type=float, default=1, help="Request arrival rate."
    )
    parser.add_argument(
        "--distro",
        type=str,
        choices=['R', 'L', 'N', 'U', 'B'],
        help='R=short, L=long, N=normal, U=uniform, B=bimodal',
    )
    parser.add_argument(
        "--msg",
        type=str,
        default='final_tput'
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--exp-json", type=str, required=True,
    )
    parser.add_argument(
        "--no-api", action='store_true',
        help='Run non-api bench, this strictly follows the order of input json'
    )
    parser = EngineArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    # mu, sigma = args.api_time_miu, args.api_time_sig
    # if args.log:
    #     log_path = args.log
    Path(f'./{args.msg}').mkdir(parents=True, exist_ok=True)
    main(args)
    
    
