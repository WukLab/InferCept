from typing import List, Optional, Tuple, Dict
import argparse
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from vllm import EngineArgs, LLMEngine, SamplingParams, utils
import torch
import json
import random
import os
from vllm.outputs import RequestOutput

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def api_call(input: str):
    return " a "

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, List[int], int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep long prompts
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) 
               for data in dataset if len(data["conversations"][0]["value"]) >= 300]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 100 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

def parse_ref_outputs() -> Dict[int, Tuple[List[int], List[int]]]:
    with open('ref-outputs.json') as f:
        outputs = json.load(f)
    ref_outputs = {data["id"]: (data["prompt_tokens"], data["output_tokens"]) for data in outputs}
    return ref_outputs

test_until = 100

def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    if args.mode == "ref":
        # Test the following prompts.
        datasets = sample_requests("ShareGPT_V3_unfiltered_cleaned_split.json", 100, engine.tokenizer)
        request_id = 0
        for prompt, prompt_token_ids, output_len in datasets:
            if request_id <= test_until:
                sampling_params = SamplingParams(n=1, temperature=0.0, presence_penalty=0.0,max_tokens=output_len,ignore_eos=True)
                engine.add_request(str(request_id), prompt, sampling_params, prompt_token_ids)
                request_id += 1
        
        outputs: List[RequestOutput] = []
        while True:
            request_outputs = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    outputs.append(request_output)
                    # print(request_output.outputs[0].token_ids)
            if not engine.has_unfinished_requests():
                break
        
        serialize = []
        outputs = sorted(outputs, key=lambda x: x.request_id)
        for output in outputs:
            serialize.append({
                "id": output.request_id,
                "prompt_tokens": output.prompt_token_ids,
                "output_tokens": output.outputs[0].token_ids,
                "output_text": output.outputs[0].text,
            })
        with open("ref-outputs.json", "w") as f:
            json.dump(serialize, f, separators=(",", ": "))
    else:
        ref_outputs = parse_ref_outputs()
        for request_id, (prompt_token_ids, output_token_ids) in ref_outputs.items():
            # if int(request_id) <= 22:
            sampling_params = SamplingParams(n=1, temperature=0.0, presence_penalty=0.0,max_tokens=len(output_token_ids),ignore_eos=True)
            engine.add_request(request_id, "", sampling_params, prompt_token_ids)
            
        outputs: List[RequestOutput] = []
        while True:
            request_outputs = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    outputs.append(request_output)
                    # print(request_output.outputs[0].token_ids)
            if not engine.has_unfinished_requests():
                break
        print(f'finished {len(outputs)} requests')
        serialize = []
        outputs = sorted(outputs, key=lambda x: x.request_id)
        for output in outputs:
            serialize.append({
                "id": output.request_id,
                "prompt_tokens": output.prompt_token_ids,
                "output_tokens": output.outputs[0].token_ids,
                "output_text": output.outputs[0].text,
            })
        with open("test-outputs.json", "w") as f:
            json.dump(serialize, f, separators=(",", ": "))
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument('--mode', type=str, default="test", choices=["ref", "test"])
    args = parser.parse_args()
    main(args)
