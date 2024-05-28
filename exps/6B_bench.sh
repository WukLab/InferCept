#!/bin/bash

# 6B
sed -i '/# replace t_sin$/c\        a = 0.00462 # replace t_sin' ../vllm/core/scheduler_v2.py
sed -i '/# replace t_offset$/c\        b = 108.99 # replace t_offset' ../vllm/core/scheduler_v2.py
sed -i '/# replace f_ch a$/c\        a = 0.0463 # replace f_ch a' ../vllm/core/scheduler_v2.py
sed -i '/# replace f_ch c$/c\        c = 10 # replace f_ch c' ../vllm/core/scheduler_v2.py

# vanilla vLLM
sed -i '/# switch it for vanilla discard$/c\            self.waiting.append(seq_group) # switch it for vanilla discard' ../vllm/core/scheduler.py
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.0 --msg results/6B_1.0_1800_vllm --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.5 --msg results/6B_1.5_1800_vllm --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.0 --msg results/6B_2.0_1800_vllm --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.5 --msg results/6B_2.5_1800_vllm --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/6B_3.0_1800_vllm --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.5 --msg results/6B_3.5_1800_vllm --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/6B_4.0_1800_vllm --window 1800 --api-policy D --exp-json merged_exp_uniform.json

# improved discard
sed -i '/# switch it for vanilla discard$/c\            self.waiting.insert(0, seq_group) # switch it for vanilla discard' ../vllm/core/scheduler.py
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.0 --msg results/6B_1.0_1800_discard --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.5 --msg results/6B_1.5_1800_discard --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.0 --msg results/6B_2.0_1800_discard --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.5 --msg results/6B_2.5_1800_discard --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/6B_3.0_1800_discard --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.5 --msg results/6B_3.5_1800_discard --window 1800 --api-policy D --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/6B_4.0_1800_discard --window 1800 --api-policy D --exp-json merged_exp_uniform.json

# swap
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.0 --msg results/6B_1.0_1800_swap --window 1800 --api-policy S --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.5 --msg results/6B_1.5_1800_swap --window 1800 --api-policy S --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.0 --msg results/6B_2.0_1800_swap --window 1800 --api-policy S --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.5 --msg results/6B_2.5_1800_swap --window 1800 --api-policy S --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/6B_3.0_1800_swap --window 1800 --api-policy S --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.5 --msg results/6B_3.5_1800_swap --window 1800 --api-policy S --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/6B_4.0_1800_swap --window 1800 --api-policy S --swap-space 128 --exp-json merged_exp_uniform.json


# preserve
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.0 --msg results/6B_1.0_1800_preserve --window 1800 --api-policy P --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.5 --msg results/6B_1.5_1800_preserve --window 1800 --api-policy P --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.0 --msg results/6B_2.0_1800_preserve --window 1800 --api-policy P --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.5 --msg results/6B_2.5_1800_preserve --window 1800 --api-policy P --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/6B_3.0_1800_preserve --window 1800 --api-policy P --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.5 --msg results/6B_3.5_1800_preserve --window 1800 --api-policy P --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/6B_4.0_1800_preserve --window 1800 --api-policy P --exp-json merged_exp_uniform.json

# InferCept
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.0 --msg results/6B_1.0_1800_V --window 1800 --api-policy V --chunk-fill --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 1.5 --msg results/6B_1.5_1800_V --window 1800 --api-policy V --chunk-fill --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.0 --msg results/6B_2.0_1800_V --window 1800 --api-policy V --chunk-fill --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 2.5 --msg results/6B_2.5_1800_V --window 1800 --api-policy V --chunk-fill --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.0 --msg results/6B_3.0_1800_V --window 1800 --api-policy V --chunk-fill --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 3.5 --msg results/6B_3.5_1800_V --window 1800 --api-policy V --chunk-fill --swap-space 128 --exp-json merged_exp_uniform.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_real.py --load-format dummy --qps 4.0 --msg results/6B_4.0_1800_V --window 1800 --api-policy V --chunk-fill --swap-space 128 --exp-json merged_exp_uniform.json
