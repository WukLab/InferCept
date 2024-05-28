"""Benchmark offline inference throughput."""
import argparse
import csv
from datetime import datetime
import json
import random
import time
from typing import List, Optional, Tuple, Dict

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

mu, sigma = 1, 0
log_path = './api_simu.log'

def save_to_csv(args, total_tks, total_time, throughput):
    print("Breakpoint")
    with open(args.log_path, "a+", newline='') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write headers if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(['Timestamp', 'Tokens', 'Total Time(s)', 'Total Throughput', 'Experiment CSV', 'Api Policy', 'Input Length', 'Output Length', 'Num Prompts', 'API Time MIU', 'API Time SIG', 'API Return Length', 'API Max Calls', 'API Inv Offset', 'API Inv Mod', 'Model', 'Tokenizer', 'Revision', 'Tokenizer Revision', 'Tokenizer Mode', 'Trust Remote Code', 'Download Dir', 'Load Format', 'Data Type', 'Max Model Length', 'Worker Use Ray', 'Pipeline Parallel Size', 'Tensor Parallel Size', 'Block Size', 'Seed', 'Swap Space', 'GPU Memory Utilization', 'Max Num Batched Tokens', 'Max Num Seqs', 'Disable Log Stats', 'Quantization', 'Chunk Fill', 'Chunk Size', 'API Policy', 'Heuristic Coef', 'Discard Policy', 'Resize Model', 'N Layer', 'N Embed', 'N Head'])

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write data to CSV
        csvwriter.writerow([
            timestamp,
            total_tks,
            total_time,
            throughput,
            args.mixed_csv,
            args.api_policy,

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
            args.block_size,
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

    # configs
    config_list = []
    if args.mixed:
        configs = csv.DictReader(open(args.mixed_csv))
        for row in configs:
            new_row = {} 
            for k,v in row.items():
                new_row[k] = float(v)
            config_list.append(new_row)
        
        # generate indices
        indices = [random.randint(0, len(config_list)-1) for _ in range(args.num_prompts)]
        with open("run2.log", "a+") as logfile: 
            logfile.write(f'{indices}')
    
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
            stop=stop,
            use_api_simulator=True,
            api_return_length=int(api_ret_len),
            api_invoke_interval=int(api_inv_offset) + request_id % int(api_inv_mod),
            api_exec_time=api_exec_time,
            api_max_calls=int(api_max_calls)
        )
        engine.add_request(
            request_id=str(request_id),
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
    pbar = tqdm(total=args.num_prompts, desc="Processed prompts")
    start = time.perf_counter()
    # Run the engine.
    outputs: List[RequestOutput] = []
    iter = 0
    torch.cuda.cudart().cudaProfilerStart()
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)
            if output.paused:
                # print(f'iter: {iter}, output: {output}')
                sampling_params: SamplingParams = engine.scheduler.paused[output.request_id][0].sampling_params
                for (rid, sid) in output.paused:
                    task = api_engine.add_task(output.request_id, sid, sampling_params.api_exec_time, sampling_params.api_return_length)
                    tasks.add(task)
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
    # for request_output in outputs:
    #     for seq_output in request_output.outputs:
    #         print(seq_output.text)
    #         print(seq_output.token_ids)
    print(f'total tokens: {engine.scheduler.total_tks}')
    return end - start, engine.scheduler.total_tks

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    elapsed_time, total_tks = run_vllm(
        args,
    )
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in CACHE_EVENTS])
    print(f'cache events time, mean: {times.mean():<10} max: {times.max():<10} min: {times.min():<10} total:  {times.sum():<10}')
    total_num_tokens = (args.input_len + args.output_len) * args.num_prompts
    save_to_csv(args, total_tks, elapsed_time, total_num_tokens / elapsed_time)
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
        "--mixed", action='store_true', help="Mixed batch"
    )
    parser.add_argument(
        "--log", type=str, default=None,
    )
    parser.add_argument(
        "--mixed-csv", type=str, default="config.csv"
    )
    parser = EngineArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    mu, sigma = args.api_time_miu, args.api_time_sig
    if args.log:
        args.log_path = args.log
    main(args)
    
