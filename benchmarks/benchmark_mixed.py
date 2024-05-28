import argparse
import json
import random
import time
from typing import List, Tuple, Dict
import torch

from vllm import EngineArgs, LLMEngine, SamplingParams, utils, LLM
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from vllm.core.scheduler import PreemptionMode

# testing workload of mixed normal generation and API(multi token)
# label: |	 gen 1  |   gen 2  |    mul 1  |
# query: |         -|		  -|       ----|
# kvs:   |----------|----------|-----------|

def main(args: argparse.Namespace):
	num_seqs = args.num_prompt + args.num_gen + args.num_mul
	# Parse the CLI argument and initialize the engine.
	llm = LLM(
		model=args.model,
		tokenizer=args.tokenizer,
		tensor_parallel_size=args.tensor_parallel_size,
		max_num_seqs=256,
		max_num_batched_tokens=None,
		trust_remote_code=args.trust_remote_code,
		disable_log_stats=args.disable_log_stats,
		swap_space=args.swap_space,
		api_policy=args.api_policy,
		heuristic_coef=args.heuristic_coef,
		load_format=args.load_format,
		resize_model=args.resize_model,
		n_layer=args.n_layer,
		n_embed=args.n_embed,
		n_head=args.n_head
	)
	stop = utils.get_api_stop_strings()
	mul_sampling_params = SamplingParams(
		n=1,
		temperature=0.0,
		top_p=1.0,
		use_beam_search=False,
		ignore_eos=True,
		max_tokens=2*args.ctx_len,
		stop=stop,
		use_api_simulator=True,
		api_return_length=1, #dummy
		api_invoke_interval=args.ctx_len-args.mul_qs,
		api_exec_time=1000,
  		api_max_calls=1
	)
	print(mul_sampling_params)
	gen_sampling_params = SamplingParams(
		n=1,
		temperature=0.0,
		top_p=1.0,
		use_beam_search=False,
		ignore_eos=True,
		max_tokens=2*args.ctx_len,
		stop=stop,
		use_api_simulator=True,
		api_return_length=1, #dummy
		api_invoke_interval=args.ctx_len-1,
		api_exec_time=0,
  		api_max_calls=1
	)

	# print("Warming up...")
	# dummy_prompt_token_ids = [[0] * args.ctx_len] * num_seqs 
	# warmup_sampling_params = SamplingParams(
	# 	n=1,
	# 	temperature=0.0,
	# 	top_p=1.0,
	# 	use_beam_search=False,
	# 	ignore_eos=True,
	# 	max_tokens=128
	# )
   
	# llm.generate(prompt_token_ids=dummy_prompt_token_ids,
	# 			sampling_params=warmup_sampling_params,
	# 			use_tqdm=False)
	eng = llm.llm_engine
 
	# Run the engine by calling `engine.step()` manually.
	gen_reqs = [[0] * 1] * args.num_gen
	mul_reqs = [[0] * 1] * args.num_mul
 
	def _add_req(seqs: List[List[int]], msg: str, sampling_params: SamplingParams):
		for i, token_ids in enumerate(seqs):
			eng.add_request(
				msg + str(i),
				prompt=None,
				sampling_params=sampling_params,
				prompt_token_ids=token_ids)
   
	_add_req(gen_reqs, "gen", gen_sampling_params)
	_add_req(mul_reqs, "mul", mul_sampling_params)
	
	# remove profile run synchronize
	# first step for prompt before entering normal generation
	while len(eng.scheduler.paused) < args.num_gen + args.num_mul:
		eng.step()
	pauesed_groups = list(eng.scheduler.paused.keys())
	print(f'paused groups: {len(pauesed_groups)}')
	for req_id in pauesed_groups:
		seq_group, _ = eng.scheduler.get_paused_seq_group(req_id)
		response = {}
		api_ret = [] if req_id.startswith("gen") else [0] * (args.mul_qs - 1)
		response[seq_group.get_seqs()[0].seq_id] = api_ret
		eng.resume_request(req_id, response)
  
	# add prompt workload
	if args.num_prompt > 0 and args.num_gen + args.num_mul > 0:
		raise ValueError("Prompt workload must be run alone.")
	prompt_reqs = [[0] * args.ctx_len] * args.num_prompt
	_add_req(prompt_reqs, "prompt", gen_sampling_params)
 
	torch.cuda.cudart().cudaProfilerStart()
	(seq_group_metadata_list, scheduler_outputs) = eng.scheduler.schedule(eng.paused_swap_out)
	assert len(seq_group_metadata_list) == num_seqs, f'{len(seq_group_metadata_list)}'
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	start.record()
	# outputs = llm.llm_engine.step()
	# break down to only time model forward and sampler
	output = eng._run_workers(
		"execute_model",
		seq_group_metadata_list=seq_group_metadata_list,
		blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
		blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
		blocks_to_copy=scheduler_outputs.blocks_to_copy,
	)
	# torch.cuda.synchronize()
	end.record()
	# ensure that all seqs will be stepped
	assert(len(output) == num_seqs)
	torch.cuda.synchronize()
	ms = start.elapsed_time(end)
	torch.cuda.cudart().cudaProfilerStop()
	print(f'Latency: {ms} ms')
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='performance of mixed batch')
	parser = EngineArgs.add_cli_args(parser)
	parser.add_argument("--num-prompt", type=int, default=0,
						help="Number of prompts to process.")
	parser.add_argument("--num-gen", type=int, default=0,
						help="Number of normal decoding.")
	parser.add_argument("--num-mul", type=int, default=1,
                     	help="Number of multi token decoding.")
	
	parser.add_argument('--ctx-len', type=int, default=128)
	parser.add_argument('--mul-qs', type=int, default=16)# increase 
	

	args = parser.parse_args()
	num_seqs = args.num_prompt + args.num_gen + args.num_mul
	print(f'num sequences: {num_seqs}\n',
       	  f'context length: {args.ctx_len}\n',
          f'num prompt: {args.num_prompt}\n',
          f'num gen: {args.num_gen}\n',
          f'num mul: {args.num_mul}\n',
          f'queries per mul: {args.mul_qs}')
	main(args)