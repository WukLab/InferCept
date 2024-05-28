# Input file
# Output file
import argparse
import subprocess
from rich import print
import os
import time
def main(output_file, experiment_file, cuda_device, num_prompts):
    # for api_policy in ['D', 'P', 'S', 'H']:
    # for api_policy in ['G', 'D', 'P', 'H-S', 'H-D', 'H-B', 'S']:
    for api_policy in ['D', 'V']:
        print(f"[red] Running Api Policy {api_policy} [/red]")
        script = f"CUDA_VISIBLE_DEVICES={cuda_device} python3 benchmarks/final_tput_bench_real.py --model EleutherAI/gpt-j-6b --load-format dummy --exp-json {experiment_file} --msg {output_folder} --api-policy {api_policy} --window 600 --qps 1 --distro"
        if api_policy == "G":
            script += " --chunk-fill "
        if api_policy in ['G', 'H-S', 'H-B', 'S']:
            script += " --swap-space 256 "
        print(script)
        with open("exp.log", "a") as f:
            f.write(f"{script}\n")
        subprocess.run(script, shell=True)

def run_single():
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Add command-line arguments with default values
    parser.add_argument("--output_file", type=str, default="exp_logs/basic.csv", help="Path to the output file")
    parser.add_argument("--experiment_file", type=str, default="config.csv", help="Path to the experiment file")
    parser.add_argument("--cuda_device", type=int, default=4, help="CUDA device number")
    parser.add_argument("--num_prompts", type=int, default=1, help="Number prompts")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.output_file, args.experiment_file, args.cuda_device, args.num_prompts)

if __name__ == "__main__":
    run_single()
