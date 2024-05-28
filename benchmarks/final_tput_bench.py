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
from vllm.outputs import RequestOutput
import numpy as np

log_path = './final_tput_results.log'
dump_path = './final_tput_dump.log'

class Request:
    def __init__(self, request_id, prompt, sampling_params, prompt_token_ids):
        self.request_id = request_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.prompt_token_ids = prompt_token_ids

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
    
    def resume(self, vllm_engine: LLMEngine, num_apis, latest_api_resume, start_measure) -> None:
        api_rets = self._get_results()
        resume_time = time.perf_counter()
        for request_id, seq_id_to_ret_len in api_rets.items():
            response = {}
            if start_measure:
                # num_apis[request_id] = (num_apis[request_id][0]+1, num_apis[request_id][1])
                # latest_api_resume[request_id] = time.perf_counter()
                pause_time = num_apis[request_id][-1][0]
                num_apis[request_id][-1] = (pause_time, resume_time)
            for seq_id, ret_len in seq_id_to_ret_len.items():
                response[seq_id] = [0] * ret_len
            vllm_engine.resume_request(request_id, response)
    
    def generate_exec_times(self, distro, num_prompts):
        rng = np.random.default_rng()
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
    stop = [utils.get_api_stop_string()]
    api_engine = APIExecutor()
    tasks = set()
    
    dummy_prompt_token_ids = [[0] * args.input_len] * args.num_prompts
    api_exec_times = api_engine.generate_exec_times(args.distro, args.num_prompts)

    requests: List[Request] = []
    rng = np.random.default_rng()
    arrival_times = []
    start_offset = 0
    num_apis = {}

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
        num_apis[str(request_id)] = []

    pbar = tqdm(total=args.num_prompts, desc="Processed prompts")
    start = time.perf_counter()

    for r in requests[:int(args.qps)+1]:
        engine.add_request(r.request_id, r.prompt, r.sampling_params, r.prompt_token_ids)
        arrival_times.pop(0)
    requests = requests[int(args.qps)+1:]

    # track latency
    start_times = {}
    end_times = {}
    latest_api_resume = {}

    # Run the engine.
    outputs: List[RequestOutput] = []
    iter = 0
    torch.cuda.cudart().cudaProfilerStart()
    tokens_after_start = 0
    tokens_after_stop = 0
    start_measure_time = 0
    num_reqs = 0
    started_measure = False 
    cpu_full = False
    while engine.has_unfinished_requests() or arrival_times:
        if engine.cpu_full:
            cpu_full = True
        curr_time = time.perf_counter()
        while start+arrival_times[0] <= curr_time:
            r = requests.pop(0)
            engine.add_request(r.request_id, r.prompt, r.sampling_params, r.prompt_token_ids)
            arrival_times.pop(0)
            if started_measure:
                start_times[r.request_id] = (curr_time, 0)
        
        if not engine.has_unfinished_requests:
            api_engine.resume(engine)
            continue

        step_outputs = engine.step()
        can_stop = False
        if not started_measure and curr_time-start > 30:
            tokens_after_start = engine.scheduler.get_tokens_have_seen() + sum(len(output.prompt_token_ids) + len(output.outputs[0].token_ids) for output in outputs)
            start_measure_time = curr_time
            num_reqs = len(outputs)
            started_measure = True
        for output in step_outputs:
            if output.request_id in start_times:
                end_times[output.request_id] = (curr_time, len(output.outputs[0].token_ids))
                # print("end", end_times[output.request_id])
            else:
                if started_measure:
                    start_times[output.request_id] = (curr_time, len(output.outputs[0].token_ids))
                    # print(f"start {output.request_id}", start_times[output.request_id])
            if output.finished:
                outputs.append(output)
                    # if len(end_times) == 100:
                    #     can_stop = True
                pbar.update(1)
            if output.paused:
                # print(f'iter: {iter}, output: {output}')
                sampling_params: SamplingParams = engine.scheduler.paused[output.request_id][0].sampling_params

                num_apis[output.request_id].append((curr_time, curr_time))
                
                for (rid, sid) in output.paused:
                    task = api_engine.add_task(output.request_id, sid, sampling_params.api_exec_time, sampling_params.api_return_length)
                    tasks.add(task)
        api_engine.resume(engine, num_apis, latest_api_resume, started_measure)
        iter += 1
        if curr_time-start > 1830:
            can_stop = True
            tokens_after_stop = engine.scheduler.get_tokens_have_seen() + sum(len(output.prompt_token_ids) + len(output.outputs[0].token_ids) for output in outputs)
            num_reqs = len(outputs) - num_reqs
            # print(tokens_after_stop)
        if can_stop:
            break
    torch.cuda.cudart().cudaProfilerStop()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    end = time.perf_counter()
    pbar.close()
    actual_tokens = tokens_after_stop - tokens_after_start

    # parse latencies
    latencies = []
    for k,v in end_times.items():
        end_time, end_length = v
        start_time, start_length = start_times[k]

        api_time = 0
        apis_list = num_apis[k]
        while apis_list:
            api_start, api_end = apis_list.pop(0)
            bumped_start = max(api_start, start_time)
            bumped_end = min(api_end, end_time)
            api_time += bumped_end - bumped_start
        
        # num_apis_for_item, api_time_for_item = num_apis[k]
        # api_time = num_apis_for_item * api_time_for_item
        # print(f"api time: {api_time}")
        # if k in latest_api_resume:
        #     latest_api_resume_for_item = latest_api_resume[k]
        #     print(f"latest resume: {latest_api_resume_for_item}")
        #     if latest_api_resume_for_item > end_time:
        #         api_time -= api_time_for_item
        #         print(f"new api time: {api_time}")
        #     if latest_api_resume_for_item - start_time 
        # print(f"end - start time - api time: {end_time}-{start_time}-{api_time} = {end_time-start_time-api_time}")
        # print(f"end - start length: {end_length}-{start_length} = {end_length-start_length}")
        final_norm_latency = (end_time-start_time-api_time)/(end_length-start_length)
        # print(final_norm_latency)
        latencies.append(final_norm_latency)
    
    # print(latencies)
    # print(np.mean(latencies))
    # print(np.median(latencies))

    return end - start_measure_time, engine.scheduler.total_tks, actual_tokens, np.median(latencies), cpu_full, num_reqs


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    elapsed_time, total_tks, actual_tks, normalized_latency, cpu_full, num_reqs = run_vllm(
        args,
    )
    
    torch.cuda.synchronize()
    with open(log_path, "a+") as logfile: 
        logfile.write(
            f"{args.api_policy},{args.distro},{args.qps},{num_reqs / elapsed_time:.4f},{normalized_latency:.4f},\n"
        )
    with open(dump_path, "a+") as logfile: 
        logfile.write(
            f"###### RUN ########\n"
            f'actual tokens: {actual_tks}\n'
            f'total tokens: {total_tks}\n'
            f'distribution: {args.distro}\n'
            f'args: {args}\n'
            f'time: {elapsed_time:.2f} s, \n'
            f"{actual_tks / elapsed_time:.2f} tokens/s\n"
            f"Was CPU over 98% utilized: {cpu_full}\n"
            f"###### RUN ########\n"
        )
    print(f"{args.api_policy},{args.distro},{num_reqs / elapsed_time:.4f},{total_tks},{normalized_latency}\n")


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
    main(args)
    
    
