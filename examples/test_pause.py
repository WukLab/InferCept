import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams, utils
import torch
import os
from typing import List, Optional, Tuple, Dict
from vllm.outputs import RequestOutput
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def api_call(input: str):
    return " a "

def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    stop = [utils.get_api_stop_string()] if args.test else None
    # Test the following prompts.
    test_prompts = [
        ("Imagine a futuristic city in the year 2150, where advanced technology and environmental sustainability are perfectly integrated. Describe this city in great detail, focusing on aspects such as architecture, transportation, energy sources, and the daily lives of its inhabitants. How do the buildings look, and what innovative materials are they made from? Describe the public transportation system and how it differs from systems in the early 21st century. What are the primary energy sources, and how are they harnessed and distributed? How do the residents of this city work, entertain themselves, and interact with technology in their everyday lives? In addition, consider the city's government and societal structure. How is the city governed, and what kind of political system is in place? What are the core values and principles that guide decision-making? Discuss how this city ensures the well-being of its citizens, including healthcare, education, and social services. How does this city handle issues like crime, conflict resolution, and the preservation of civil liberties? Furthermore, explore the relationship of this city with the natural environment. How does the city maintain a balance with nature, and what are its strategies for conservation and biodiversity? Are there any unique parks, green spaces, or integration of natural elements within the urban landscape? Lastly, imagine a scenario where this city faces a significant challenge, such as a natural disaster or a technological crisis. How does the city respond and recover from this event? What systems and protocols are in place to handle such emergencies, and what role do citizens play in these situations? Please provide a comprehensive and imaginative description of each of these aspects, creating a vivid and detailed portrayal of life in this futuristic city.",
        SamplingParams(n=1, temperature=0.0, presence_penalty=0.0,stop=stop,max_tokens=100)),
        # ("Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...",
        #  SamplingParams(n=1, temperature=0.0, presence_penalty=0.2,stop=stop,max_tokens=100)),
    ] * 100

    # Run the engine by calling `engine.step()` manually.
    request_id = 0
    # To test iteration-level scheduling, we add one request at each step.
    for prompt, sampling_params in test_prompts:
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1
    
    outputs: List[RequestOutput] = []
    if not args.test:
        while True:
        # for _ in range(29):
            request_outputs = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    outputs.append(request_output)
                    # print(request_output.outputs[0].token_ids)
            if not engine.has_unfinished_requests():
                break
    else:
        # TO the test, only pause once
        torch.cuda.cudart().cudaProfilerStart()
        while True:
            request_outputs = engine.step()
            for request_output in request_outputs:
                # print(request_output)
                if request_output.finished:
                    outputs.append(request_output)
                    # print(request_output.outputs[0].token_ids)
                if request_output.paused:
                    response = {}
                    for (rid, sid) in request_output.paused:
                        # response[sid] = [582, 508, 468, 587]
                        # response[sid] = [198, 464, 13429, 21983, 25, 198, 198, 818, 262, 614, 2310, 1120, 11, 257]
                        # normal_ret
                        response[sid] = [50118, 133, 511, 16, 10, 889, 9, 5, 144, 505, 5894, 9, 5, 343, 35, 50118, 50118, 134, 4, 20, 343, 18, 2112, 50118, 50118, 133]
                    engine.resume_request(request_output.request_id, response)
            if not engine.has_unfinished_requests():
                break
        torch.cuda.cudart().cudaProfilerStop()
        
    print(f'finished {len(outputs)} requests')
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    serialize = []
    for output in outputs:
        serialize.append({
            "id": output.request_id,
            "output_tokens": output.outputs[0].token_ids,
            "output_text": output.outputs[0].text,
        })
    with open(f'pause_{args.test}_output.json', 'w+') as f:
        json.dump(serialize, f, separators=(",", ": "))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)
