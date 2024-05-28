#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu_naive.py --api-time-miu 60.0 --api-ret-len 16 --api-max-calls 1 --api-inv-offset 128 --api-inv-mod 128 
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu_naive.py --api-time-miu 60.0 --api-ret-len 128 --api-max-calls 1 --api-inv-offset 128 --api-inv-mod 128 
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu_naive.py --api-time-miu 60.0 --api-ret-len 512 --api-max-calls 1 --api-inv-offset 128 --api-inv-mod 128 
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu_naive.py --api-time-miu 60.0 --api-ret-len 64 --api-max-calls 4 --api-inv-offset 128 --api-inv-mod 64 
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu_naive.py --api-time-miu 60.0 --api-ret-len 256 --api-max-calls 4 --api-inv-offset 0 --api-inv-mod 1 
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_api_simu_naive.py --api-time-miu 60.0 --api-ret-len 64 --api-max-calls 8 --api-inv-offset 64 --api-inv-mod 1 
