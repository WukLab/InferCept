#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu.py --api-time-miu 10 --api-ret-len 128 --api-max-calls -1 --api-inv-offset 128 --api-inv-mod 128 --num-prompts 500
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu.py --api-time-miu 10 --api-ret-len 128 --api-max-calls -1 --api-inv-offset 128 --api-inv-mod 64 --num-prompts 500
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu.py --api-time-miu 10 --api-ret-len 128 --api-max-calls -1 --api-inv-offset 128 --api-inv-mod 32 --num-prompts 500
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu.py --api-time-miu 10 --api-ret-len 128 --api-max-calls -1 --api-inv-offset 128 --api-inv-mod 16 --num-prompts 500
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu.py --api-time-miu 10 --api-ret-len 128 --api-max-calls -1 --api-inv-offset 128 --api-inv-mod 8 --num-prompts 500