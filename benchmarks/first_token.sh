#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python benchmark_api_simu.py --api-time-miu 0.1 --api-ret-len 16 --api-max-calls 8 --api-inv-offset 64 --api-inv-mod 48 --num-prompts 500
CUDA_VISIBLE_DEVICES=0 python benchmark_api_simu.py --api-time-miu 0.1 --api-ret-len 128 --api-max-calls 8 --api-inv-offset 0 --api-inv-mod 1 --num-prompts 500
CUDA_VISIBLE_DEVICES=0 python benchmark_api_simu.py --api-time-miu 30 --api-ret-len 16 --api-max-calls 8 --api-inv-offset 64 --api-inv-mod 48 --num-prompts 500
CUDA_VISIBLE_DEVICES=0 python benchmark_api_simu.py --api-time-miu 30 --api-ret-len 128 --api-max-calls 8 --api-inv-offset 0 --api-inv-mod 1 --num-prompts 500