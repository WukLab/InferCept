"""Benchmark offline inference throughput."""
import argparse
import random
import time
from typing import List, Dict

import torch
import queue
import threading
from tqdm.rich import tqdm

from vllm import SamplingParams, LLMEngine, EngineArgs, utils
from vllm.model_executor.layers.attention import CACHE_EVENTS
from vllm.outputs import RequestOutput
import numpy as np
from pathlib import Path

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
            vllm_engine.resume_request(request_id, response)
    
    def generate_exec_times(self, distro, num_prompts, seed):
        rng = np.random.default_rng(seed)
        if distro == 'N':
            # normal distribution
            return np.abs(rng.normal(loc=6, scale=2, size=(num_prompts,)))
        elif distro == 'U':
            # uniform distribution
            return rng.uniform(low=0.1, high=15, size=(num_prompts,))
        else:
            # Generate random numbers from gamma distribution
            right = np.abs(rng.gamma(shape=0.5, scale=4, size=(num_prompts,)))  # shorter api times
            left = np.abs(20-right)                                             # longer api times
            if distro == 'L':
                return left
            elif distro == 'R':
                return right
            elif distro == 'B':
                return np.concatenate((rng.choice(left, num_prompts//2),
                                       rng.choice(right, num_prompts//2 + 1)))
            else:
                return ValueError(f'Unsupported distribution: {distro}')

def run_vllm(
    args: argparse.Namespace,
) -> float:
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    stop = [utils.get_api_stop_string()]
    api_engine = APIExecutor()
    tasks = set()
    args.num_prompts = int(args.qps * args.window) + 1
    dummy_prompt_token_ids = [[0] * args.input_len] * args.num_prompts
    api_exec_times = api_engine.generate_exec_times(args.distro, args.num_prompts, args.seed)

    requests: List[Request] = []
    rng = np.random.default_rng()
    arrival_times = []
    start_offset = 0
    #num_apis = {}

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
            api_return_length=int(args.api_ret_len),
            api_invoke_interval=int(args.api_inv_offset) + request_id % int(args.api_inv_mod), # Completion tokens after the api calls finishes. Prompt/final
            api_exec_time=api_exec_times[request_id],
            api_max_calls=int(args.api_max_calls)
        )
        requests.append(Request(
                request_id=str(request_id),
                prompt=None,
                sampling_params=sampling_params,
                prompt_token_ids=prompt_token_ids
            )
        )
        offset = rng.exponential(1.0/args.qps)
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
    
    if CACHE_EVENTS:
        times = []
        with open(f'./{args.msg}/swap_wait.csv', 'w+') as f:
            for s, e, t, i, o in CACHE_EVENTS:
                times.append(s.elapsed_time(e))
                f.write(f'{times[-1]}, {t}, {i}, {o}\n')
        print(f'cache events time mean: {np.mean(times):<10} max: {np.max(times):<10} min: {np.min(times):<10} total:  {np.sum(times):<10}')
        
    with open(f'./{args.msg}/paused_mem.log', 'w+') as w:
        for b in engine.scheduler.paused_gpu_blocks:
            w.write(f'{b}\n')

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
            
            final_norm_latency = (end_time-start_time-api_time)/(end_length-start_length)
            # final_norm_latency = (end_time-start_time)/(end_length-start_length)
            latencies.append(final_norm_latency)
            t.write(f'{r.request_id}, {r.finished}, {api_exec_times[int(r.request_id)]}, {r.end_time-r.start_time}, {api_time}, {start_length}, {end_length}, {final_norm_latency}\n')

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
            f'time: {elapsed_time:.2f} s, \n'
            f"{actual_tks / elapsed_time:.2f} tokens/s\n"
            f"finished: {num_reqs}\n"
            f"Was CPU over 98% utilized: {cpu_full}\n"
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
    # parser.add_argument(
    #     "--log", type=str, default=None,
    # )
    # parser.add_argument(
    #     "--api-time-miu", type=float, default=6, help="API execution time mean."
    # )
    # parser.add_argument(
    #     "--api-time-sig", type=float, default=2, help="API execution time sigma."
    # )
    # parser.add_argument(
    #     "--mixed", action='store_true', help="Mixed batch"
    # )
    parser = EngineArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    # mu, sigma = args.api_time_miu, args.api_time_sig
    # if args.log:
    #     log_path = args.log
    Path(f'./{args.msg}').mkdir(parents=True, exist_ok=True)
    main(args)
    
    
