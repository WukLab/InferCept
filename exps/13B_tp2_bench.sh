#!/bin/bash

# 13B - tp2
sed -i '/# replace t_sin$/c\        a = -0.3297119141 # replace t_sin' ../vllm/core/scheduler_v2.py
sed -i '/# replace t_offset$/c\        b = 93.994140625 # replace t_offset' ../vllm/core/scheduler_v2.py
sed -i '/# replace f_ch a$/c\        a = 0.0279 # replace f_ch a' ../vllm/core/scheduler_v2.py
sed -i '/# replace f_ch c$/c\        c = 15.4 # replace f_ch c' ../vllm/core/scheduler_v2.py

# vanilla vllm
sed -i '/# switch it for vanilla discard$/c\            self.waiting.append(seq_group) # switch it for vanilla discard' ../vllm/core/scheduler.py
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.0 --msg results/13B_1.0_1800_vllm_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.5 --msg results/13B_1.5_1800_vllm_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.0 --msg results/13B_2.0_1800_vllm_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.5 --msg results/13B_2.5_1800_vllm_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.0 --msg results/13B_3.0_1800_vllm_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.5 --msg results/13B_3.5_1800_vllm_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 4.0 --msg results/13B_4.0_1800_vllm_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2

# improved discard
sed -i '/# switch it for vanilla discard$/c\            self.waiting.insert(0, seq_group) # switch it for vanilla discard' ../vllm/core/scheduler.py
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.0 --msg results/13B_1.0_1800_discard_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.5 --msg results/13B_1.5_1800_discard_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.0 --msg results/13B_2.0_1800_discard_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.5 --msg results/13B_2.5_1800_discard_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.0 --msg results/13B_3.0_1800_discard_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.5 --msg results/13B_3.5_1800_discard_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 4.0 --msg results/13B_4.0_1800_discard_tp2 --window 1800 --api-policy D --exp-json merged_exp_uniform.json -tp 2

# swap
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.0 --msg results/13B_1.0_1800_swap_tp2 --window 1800 --api-policy S --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.5 --msg results/13B_1.5_1800_swap_tp2 --window 1800 --api-policy S --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.0 --msg results/13B_2.0_1800_swap_tp2 --window 1800 --api-policy S --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.5 --msg results/13B_2.5_1800_swap_tp2 --window 1800 --api-policy S --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.0 --msg results/13B_3.0_1800_swap_tp2 --window 1800 --api-policy S --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.5 --msg results/13B_3.5_1800_swap_tp2 --window 1800 --api-policy S --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 4.0 --msg results/13B_4.0_1800_swap_tp2 --window 1800 --api-policy S --swap-space 64 --exp-json merged_exp_uniform.json -tp 2

# Preserve
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.0 --msg results/13B_1.0_1800_preserve_tp2 --window 1800 --api-policy P --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.5 --msg results/13B_1.5_1800_preserve_tp2 --window 1800 --api-policy P --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.0 --msg results/13B_2.0_1800_preserve_tp2 --window 1800 --api-policy P --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.5 --msg results/13B_2.5_1800_preserve_tp2 --window 1800 --api-policy P --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.0 --msg results/13B_3.0_1800_preserve_tp2 --window 1800 --api-policy P --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.5 --msg results/13B_3.5_1800_preserve_tp2 --window 1800 --api-policy P --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 4.0 --msg results/13B_4.0_1800_preserve_tp2 --window 1800 --api-policy P --exp-json merged_exp_uniform.json -tp 2

# InferCept
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.0 --msg results/13B_1.0_1800_V_tp2 --window 1800 --api-policy V --chunk-fill --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 1.5 --msg results/13B_1.5_1800_V_tp2 --window 1800 --api-policy V --chunk-fill --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.0 --msg results/13B_2.0_1800_V_tp2 --window 1800 --api-policy V --chunk-fill --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 2.5 --msg results/13B_2.5_1800_V_tp2 --window 1800 --api-policy V --chunk-fill --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.0 --msg results/13B_3.0_1800_V_tp2 --window 1800 --api-policy V --chunk-fill --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 3.5 --msg results/13B_3.5_1800_V_tp2 --window 1800 --api-policy V --chunk-fill --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/final_tput_bench_real.py --model lmsys/vicuna-13b-v1.3 --load-format dummy --qps 4.0 --msg results/13B_4.0_1800_V_tp2 --window 1800 --api-policy V --chunk-fill --swap-space 64 --exp-json merged_exp_uniform.json -tp 2
