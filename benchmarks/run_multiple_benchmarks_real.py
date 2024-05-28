import subprocess
import threading
import queue
import time
import glob
import os
from pprint import pprint
import random

class ExperimentThread(threading.Thread):
    def __init__(self, exp_queue, result_queue, cuda_device):
        super().__init__()
        self.exp_queue = exp_queue
        self.result_queue = result_queue
        self.cuda_device = cuda_device

    def run_experiment(self, experiment):
        i = random.randint(0, 1000)

        experiment['name'] += f"-{i}"
        command = experiment["command"] + f" --cuda_device {self.cuda_device}"
        full_command = f"tmux new-session -d -s {experiment['name']} '{command}'"
        print("Running command", experiment["name"], self.cuda_device)
        status = subprocess.run(full_command, shell=True)
        # print(['tmux', 'new-session', '-d', '-s', experiment["name"], command])
        # Wait for the tmux session to finish
        while True:
            # Check if the tmux session is still running
            status = subprocess.run(['tmux', 'has-session', '-t', experiment["name"]], stdout=subprocess.PIPE)
            if status.returncode != 0:
                # Session has ended
                break

            # Sleep for a short duration before checking again
            time.sleep(1)
        print(f"The tmux session {experiment['name']} has finished.")

    def run(self):
        while True:
            experiment = self.exp_queue.get()
            if experiment is None:
                break
            self.run_experiment(experiment)
            self.exp_queue.task_done()
            self.result_queue.put(f"Experiment {experiment['name']} completed on CUDA device {self.cuda_device}")

def wait_for_sessions(experiments):
    for exp in experiments:
        subprocess.run(['tmux', 'wait-for', '-S', exp["name"]])

def main(exp_folder, log_folder, num_prompts=1, total_cuda_devices = [0, 1, 2, 3, 4]):
      # Adjust based on the number of available CUDA devices
    exp_queue = queue.Queue()
    result_queue = queue.Queue()
    threads = []
    
    exps = glob.glob(f"{exp_folder}/**/*.json", recursive=True)
    outs = [f"{log_folder}/" + exp.replace(".json", "_out.csv").replace(f"{exp_folder}/", "") for exp in exps]
    # Make directories if they don't exist
    for out in outs:
        base_name = os.path.dirname(out)
        if os.path.dirname(base_name):
            os.makedirs(base_name, exist_ok=True)
    experiments = []
    for exp, out in zip(exps, outs):
        if "_filtered" in exp:
            continue
        # if "diffusion" in exp:
        experiments.append(
            {
                "name": exp.replace("/", "_").replace(".json", ""),
                "command": f"python3 run_benchmark_api_on_all_policies.py --num_prompts {num_prompts} --experiment_file {exp} --output_file {out}",
                "exp": exp,
                "out": out
            }
        )
        #     experiments.append(
        #         {
        #             "name": exp.replace("/", "_").replace(".json", ""),
        #             "command": f"python3 run_benchmark_api_on_all_policies.py --num_prompts {num_prompts} --experiment_file {exp} --output_file {out}",
        #             "exp": exp,
        #             "out": out
        #         }
        #     )
        #     experiments.append(
        #         {
        #             "name": exp.replace("/", "_").replace(".json", ""),
        #             "command": f"python3 run_benchmark_api_on_all_policies.py --num_prompts {num_prompts} --experiment_file {exp} --output_file {out}",
        #             "exp": exp,
        #             "out": out
        #         }
        #     )

    for cuda_device in total_cuda_devices:
        thread = ExperimentThread(exp_queue, result_queue, cuda_device)
        thread.start()
        threads.append(thread)

    for exp in experiments:
        exp_queue.put(exp)

    exp_queue.join()

    for thread in threads:
        exp_queue.put(None)

    for thread in threads:
        thread.join()

    while not result_queue.empty():
        print(result_queue.get())

if __name__ == "__main__":
    # main("exp_version3", "exp_logs_version3", num_prompts=300, total_cuda_devices=[0, 1, 2, 3, 4])
    main("exp_version_mixed", "test_results", num_prompts=1000, total_cuda_devices=[5,6,7])
