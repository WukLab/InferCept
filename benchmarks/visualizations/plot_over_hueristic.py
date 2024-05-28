
import pandas as pd
import matplotlib.pyplot as plt
import os


import json
import numpy as np

import warnings

# Ignore pandas warnings related to setting values on a copy of a slice from a DataFrame
warnings.filterwarnings("ignore")

def plot_api_policy_throughput(out_folder, csv_files, extra_title=[], category_order=['H', 'W']):
    # Initialize subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()
    # Loop through each CSV file
    for i, csv_file in enumerate(csv_files):
        # Load CSV into a DataFrame
        full_path = os.path.join(out_folder, csv_file)
        df = pd.read_csv(full_path)

        # Group by API Policy and calculate mean throughput
        df['Api Policy'] = pd.Categorical(df['Api Policy'], categories=category_order, ordered=True)
        throughput_data = df.groupby('Api Policy')['Total Throughput'].mean()
        # Plot bar chart for each API Policy
        # Find the API Policy with the highest throughput
        max_throughput_policy = throughput_data.idxmax()
        speedup_data = throughput_data / throughput_data[category_order[0]]
        # Plot bar chart for each API Policy, highlight the one with the highest throughput
        bars = throughput_data.plot(kind='bar', ax=axes[i], color=['skyblue' if policy != max_throughput_policy else 'orange' for policy in throughput_data.index])
        title = f'{csv_file.split("_")[0].capitalize()} Experiment'
        if len(extra_title) > 0:
            title += f' - {extra_title[i]}'
        axes[i].set_title(title)
        axes[i].set_xlabel('API Policy')
        axes[i].set_ylabel('Throughput(Tokens/second)')
        for bar, label, speedup in zip(bars.patches, throughput_data.values, speedup_data.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{label:.2f}\nSpeedup: {speedup:.2f}', ha='center', va='bottom')

    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
    

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Calculate the PDF
def calculate_pdf(data):
    unique_values, counts = np.unique(data, return_counts=True)
    pdf = counts / len(data)
    return unique_values, pdf

# Calculate the CDF
def calculate_cdf(data):
    unique_values, counts = np.unique(data, return_counts=True)
    cdf = np.cumsum(counts) / len(data)
    return unique_values, cdf

# Plot the PDF
def plot_pdf(unique_values, pdf):
    plt.bar(unique_values, pdf, align='center', alpha=0.5)
    plt.xlabel('Values')
    plt.ylabel('PDF')
    plt.title('Probability Density Function (PDF)')
    plt.show()

# Plot the CDF
def plot_cdf(unique_values, cdf):
    plt.step(unique_values, cdf, where='post')
    plt.xlabel('Values')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.show()


# Calculate the CDF for both data sets
def calculate_cdf(data):
    unique_values, counts = np.unique(data, return_counts=True)
    cdf = np.cumsum(counts) / len(data)
    return unique_values, cdf


# Plot CDF of execution time, throughput, and latency for each experiment file
def plot_cdf_of_exec_times_calls(out_folder, json_files, extra_title=[]):
    # Initialize subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Loop through each CSV file
    for i, csv_file in enumerate(json_files):
        # Load CSV into a DataFrame
        full_path = os.path.join(out_folder, csv_file)
        with open(full_path, "r") as f:
            data = json.load(f)
            exec_times = []
            num_calls = []
            max_context_len = []
            for request_id,requests in data.items():
                request_exec_times = []
                num_calls.append(len(requests))
                for request in requests:
                    request_exec_times.append(max(0, request["api_exec_time"]))        
                # TODO get tokenizer amount
                exec_times.append(np.mean(request_exec_times))
            unique_values_gkml, cdf_gkml = calculate_cdf(exec_times)
            # Plot the CDF for both data sets
            axes[i].step(unique_values_gkml, cdf_gkml, label='GKML CDF', where='post', linestyle='-', marker='o', markersize=5, )
            axes[i].set_xlabel('Exec Time(s)')
            axes[i].set_ylabel('CDF')
            title = f'{csv_file.split("_")[0].capitalize()} Experiment'
            if len(extra_title) > 0:
                title += f' - {extra_title[i]}'
            axes[i].set_title('Cumulative Distribution Function of execution time(CDF)')
    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


    
def plot_cdf_of_props(out_folder, json_files, extra_title=[]):
    # Initialize a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    # Define color-blind-friendly colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    mean_exec_times = []
    mean_num_calls = []
    max_context_lens = []

    # Loop through each CSV file
    for i, csv_file in enumerate(json_files):
        # Load JSON into a DataFrame
        full_path = os.path.join(out_folder, csv_file)
        with open(full_path, "r") as f:
            data = json.load(f)
            exec_times = []
            num_calls = []
            max_context_len = []
            for request_id, requests in data.items():
                request_exec_times = []
                prompt_size = []
                num_calls.append(len(requests))
                for request in requests:
                    request_exec_times.append(max(0, request["api_exec_time"]))      
                    prompt_size.append(request.get("prompt_size", 0))  
                exec_times.append(np.mean(request_exec_times))
                prompt_size_total = min(np.sum(prompt_size), 2048)
                max_context_len.append(prompt_size_total) # Clip to 2048
            unique_values_gkml_exec, cdf_gkml_exec = calculate_cdf(exec_times)
            unique_values_gkml_calls, cdf_gkml_calls = calculate_cdf(num_calls)
            unique_values_context_len, cdf_gkml_context_len = calculate_cdf(max_context_len)

            # Plot the CDF for execution times on the first subplot
            mean_exec_time = np.mean(exec_times)
            mean_exec_times.append(mean_exec_time)
            ax1.step(
                unique_values_gkml_exec,
                cdf_gkml_exec,
                label=f'{csv_file.split("_")[0].capitalize()} Experiment',
                where='post',
                linestyle='-',
                marker='o',
                markersize=5,
                color=colors[i % len(colors)],
            )

            # Plot the CDF for num calls on the second subplot
            mean_num_call = np.mean(num_calls)
            mean_num_calls.append(mean_num_call)
            ax2.step(
                unique_values_gkml_calls,
                cdf_gkml_calls,
                label=f'{csv_file.split("_")[0].capitalize()} Experiment',
                where='post',
                linestyle='-',
                marker='o',
                markersize=5,
                color=colors[i % len(colors)],
            )

            mean_max_context_len = np.mean(max_context_len)
            max_context_lens.append(mean_max_context_len)
    


    # Create a legend box for mean values
    legend_text = [f'{csv_file.split("_")[0].capitalize()}: {mean:.2f}' for csv_file, mean in zip(json_files, mean_exec_times)]
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), labels=legend_text)

    legend_text = [f'{csv_file.split("_")[0].capitalize()}: {mean:.2f}' for csv_file, mean in zip(json_files, mean_num_calls)]
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), labels=legend_text)

    titles = [f'{csv_file.split("_")[0].capitalize()}' for csv_file, mean in zip(json_files, max_context_lens)]
    ax3.bar(
        range(len(max_context_lens)),
        max_context_lens,
        color=colors[0]
    )
    ax3.set_xticks(range(len(max_context_lens)), titles, rotation=45, ha='right')


    ax1.set_xlabel('Exec Time(s)')
    ax1.set_ylabel('CDF')
    ax1.set_title(f'CDF of {extra_title[0]} Execution Time')

    ax2.set_xlabel('Num Calls')
    ax2.set_ylabel('CDF')
    ax2.set_title(f'CDF of {extra_title[0]} Num Calls')

    ax3.set_xlabel('Max Context Len')
    ax3.set_ylabel('Context Len')
    ax3.set_title(f'{extra_title[0]} Context Len')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


workloads_out = {
    "slow": ["alfworld_trace_out.csv", "gkml_out.csv", "hotpot_qa_out.csv"],
    "diffusion": ["diffusion/diffusion_1_out.csv", "diffusion/diffusion_3_out.csv", "diffusion/diffusion_5_out.csv", "diffusion/diffusion_10_out.csv"],
    "tts": ["tts/tts_1_out.csv", "tts/tts_3_out.csv", "tts/tts_5_out.csv", "tts/tts_10_out.csv"],
    "web_search": ["web_search/web_search_1_out.csv", "web_search/web_search_3_out.csv", "web_search/web_search_5_out.csv", "web_search/web_search_10_out.csv"],
    "chat": ["chat/chat_bot_1_out.csv", "chat/chat_bot_3_out.csv", "chat/chat_bot_5_out.csv", "chat/chat_bot_6_out.csv",  "chat/chat_bot_10_out.csv", "chat/chat_bot_15_out.csv", "chat/chat_bot_30_out.csv"],
}

workload_exp = {
    "slow": [out.replace("csv", "json").replace("_out", "") for out in workloads_out["slow"]],
    "diffusion": [out.replace("csv", "json").replace("_out", "") for out in workloads_out["diffusion"]],
    "tts": [out.replace("csv", "json").replace("_out", "") for out in workloads_out["tts"]],
    "web_search": [out.replace("csv", "json").replace("_out", "") for out in workloads_out["web_search"]],
    "chat": [out.replace("csv", "json").replace("_out", "") + "_filtered.json" for out in workloads_out["chat"]],
}

plot_api_policy_throughput("../exp_logs_version4", workloads_out["slow"])
plot_cdf_of_props("../exp_version4", workload_exp["slow"], extra_title=["slow workload"])

plot_api_policy_throughput("../exp_logs_version4", workloads_out["diffusion"], extra_title=["Diffusion 1 call", "Diffusion 3 calls", "Diffusion 5 calls", "Diffusion 10 calls"])
plot_cdf_of_props("../exp_version4", workload_exp["diffusion"], extra_title=["diffusion workload"])
