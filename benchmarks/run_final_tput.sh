# #!/bin/bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy P --distro N --qps 40 --msg P_N_40_1000 --num-prompts 1000 &
CUDA_VISIBLE_DEVICES=1 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy D --distro N --qps 40 --msg D_N_40_1000 --num-prompts 1000 &
CUDA_VISIBLE_DEVICES=2 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy P --distro N --qps 20 --msg P_N_20_2000 --num-prompts 2000 &
CUDA_VISIBLE_DEVICES=3 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy D --distro N --qps 20 --msg D_N_20_2000 --num-prompts 2000 &
CUDA_VISIBLE_DEVICES=4 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy P --distro N --qps 40 --msg P_N_40_2000 --num-prompts 2000 &
CUDA_VISIBLE_DEVICES=5 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy D --distro N --qps 40 --msg D_N_40_2000 --num-prompts 2000 &

CUDA_VISIBLE_DEVICES=6 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy P --distro N --qps 40 --chunk-fill --msg O_N_40_1000 --swap-space 768 --num-prompts 1000
CUDA_VISIBLE_DEVICES=6 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy P --distro N --qps 20 --chunk-fill --msg O_N_20_2000 --swap-space 768 --num-prompts 2000
CUDA_VISIBLE_DEVICES=6 python benchmarks/final_tput_bench_fixed.py --load-format dummy --api-policy P --distro N --qps 40 --chunk-fill --msg O_N_40_2000 --swap-space 768 --num-prompts 2000
