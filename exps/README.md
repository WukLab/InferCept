# This is the reproduce instructions for paper "INFERCEPT: Efficient Intercept Support for Large-Language Model Inferencing"

## Dataset
Download our 6-augment mixture workload from <a href='https://drive.google.com/file/d/1CMTgd-lYFXLprKK2Q3QkCcqrXJc1UgtR/view?usp=drive_link'><b>google drive</b></a> and place it under `exps` filder.

## Profiler
The profiler is still under refactoring. The current benchmark script will set profiling variables to ones used in the paper.

## Run Benchmark
```bash
# after installing InferCept
bash bench.sh
```
1. Results will be available at `exps/results`. 
2. Each data point will run for 30min, please manage your GPU cluster wisely.
3. Please do not schedule two swap-involved run concurrently as we assume exclusive access to the PCIE bendwidth.