"""A layer that simulates the next token."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import random
from vllm import utils
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.model_executor.input_metadata import InputMetadata
from vllm.sequence import SequenceOutputs, SequenceData, SamplerOutput

DUMMY_TOKEN = 31548  # "Ġdummy"

def _greedy_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    seq_data: Dict[int, SequenceData],
) -> List[Tuple[List[int], List[int]]]:
    results = []
    for seq_group in selected_seq_groups:
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, (
            "Greedy sampling should have only one seq.")
        parent_ids = [0]
        next_token_ids = [_sample(sampling_params, seq_data[seq_ids[0]])]
        results.append((next_token_ids, parent_ids))
    return results

def _random_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    seq_data: Dict[int, SequenceData],
) -> List[Tuple[List[int], List[int]]]:
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            assert num_parent_seqs == 1, (
                "Prompt input should have only one seq.")
            parent_ids = [0] * sampling_params.best_of
            next_token_ids = [_sample(sampling_params, seq_data[seq_ids[0]])] * \
                                sampling_params.best_of
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = [_sample(sampling_params, seq_data[seq_id]) 
                                for seq_id in seq_ids]
        results.append((next_token_ids, parent_ids))
    return results

class Simulator(nn.Module):

    def forward(self,
                input_metadata: InputMetadata) -> SamplerOutput:
        categorized_seq_group_ids = {t: [] for t in SamplingType}
        category_num_tokens = {t: 0 for t in SamplingType}
        for i, seq_group in enumerate(input_metadata.seq_groups):
            seq_ids, sampling_params = seq_group
            sampling_type = sampling_params.sampling_type
            categorized_seq_group_ids[sampling_type].append(i)
            num_seqs = len(seq_ids)
            category_num_tokens[sampling_type] += num_seqs
            
        seq_outputs_dict: Dict[int, List[SequenceOutputs]] = {}
        for sampling_type in SamplingType:
            seq_group_ids = categorized_seq_group_ids[sampling_type]
            seq_groups = [input_metadata.seq_groups[i] for i in seq_group_ids]
            is_prompts = [i < input_metadata.num_prompts for i in seq_group_ids]
            num_tokens = category_num_tokens[sampling_type]
            if num_tokens == 0:
                continue
            if sampling_type == SamplingType.GREEDY:
                sample_results = _greedy_sample(seq_groups, input_metadata.seq_data)
            elif sampling_type == SamplingType.RANDOM:
                sample_results = _random_sample(seq_groups, is_prompts, input_metadata.seq_data)
            else:
                raise NotImplementedError("Beam search is not supported yet")
                
            # build output    
            for seq_group_id, seq_group, sample_result in zip(
                    seq_group_ids, seq_groups, sample_results):
                seq_ids, sampling_params = seq_group
                next_token_ids, parent_ids = sample_result
                num_results = len(next_token_ids)
                num_parent_seqs = len(seq_ids)
                seq_outputs: List[SequenceOutputs] = []
                for parent_id, next_token_id in zip(
                    parent_ids, next_token_ids):
                    seq_outputs.append(
                        SequenceOutputs(seq_ids[parent_id], next_token_id, {next_token_id: 0.0}))
                seq_outputs_dict[seq_group_id] = seq_outputs
        return [seq_outputs_dict[i] for i in range(len(input_metadata.seq_groups))]

def _sample(sampling_params: SamplingParams, seq_data: SequenceData) -> int:
    # seq_data should be updated at the master worker
    if seq_data.generation_counter == sampling_params.api_invoke_interval:
        if sampling_params.api_max_calls != 0:
            # seq_data.generation_counter = 0
            return utils.get_api_stop_token()
    # seq_data.generation_counter += 1
    return DUMMY_TOKEN