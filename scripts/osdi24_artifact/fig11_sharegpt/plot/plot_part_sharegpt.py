import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import colorsys
import argparse


request_P99_improved_ratios = []
request_mean_improved_ratios = []
prefill_P99_improved_ratios = []
prefill_mean_improved_ratios = []
decode_P99_improved_ratios = []
decode_mean_improved_ratios = []
preemption_loss_reduced_ratios = []
preemption_loss_reduced_values = []


def get_colors():
    # blue
    red, green, blue = 68, 114, 196
    color1 = (red / 255.0, green / 255.0, blue / 255.0)
    # orange
    red, green, blue = 237, 125, 49
    color2 = (red / 255.0, green / 255.0, blue / 255.0)
    # green
    red, green, blue = 112, 173, 71
    color3 = (red / 255.0, green / 255.0, blue / 255.0)

    colors = [color1, color2, color3]

    scale_factor = 10.0
    scaled_colors = []
    for color in colors:
        hsv = colorsys.rgb_to_hsv(*color)
        scaled_hsv = (hsv[0], min(hsv[1] * scale_factor, 1.0), hsv[2])
        scaled_rgb = colorsys.hsv_to_rgb(*scaled_hsv)
        scaled_colors.append(scaled_rgb)
    colors = scaled_colors

    return colors


def get_trace_data(log_path, trace):
    if trace == 'ShareGPT':
        log_trace_path = os.path.join(log_path, "sharegpt")
    else:
        log_trace_path = os.path.join(log_path, trace.replace('-', '_'))

    # Build QPS list dynamically from data discovered
    qps_list = []
    
    trace_data = {}
    metrics = ['Request P99', 'Request Mean', 'Prefill P99', 'Prefill Mean', 'Decode P99', 'Decode Mean', 'Preemption Loss']
    
    # Track which QPS values and methods we actually have data for
    available_qps = set()
    available_methods = set()
    
    for root, dirs, files in os.walk(log_trace_path):
        for file in files:
            if file.endswith('.data'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    file_data = {}
                    for line in file:
                        # Skip separators and empty lines
                        if not line.strip() or set(line.strip()) == set('-'):
                            continue
                        # Only split on the first colon to preserve timestamps like 12:18:32
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value_part = parts[1].strip()
                            # Ignore non-essential headers
                            if key in ['Conversation Mode', 'Processing Time']:
                                continue
                            if key in ['Trace', 'Method']:
                                value = value_part
                            else:
                                try:
                                    value = float(value_part)
                                except ValueError:
                                    if key not in ['QPS'] + metrics:
                                        continue
                                    print(f"Warning: Could not parse value '{value_part}' for key '{key}' in {filepath}")
                                    continue
                            file_data[key] = value
                        else:
                            continue
                    
                    if 'QPS' in file_data and 'Method' in file_data:
                        qps = file_data['QPS']
                        method = file_data['Method']
                        
                        # Only process if method is one of the expected ones
                        if method in ['Llumnix', 'All', 'All-Wait']:
                            available_qps.add(qps)
                            available_methods.add(method)
                            
                            if qps not in trace_data:
                                trace_data[qps] = {
                                    'Llumnix': {m: [] for m in metrics},
                                    'All': {m: [] for m in metrics},
                                    'All-Wait': {m: [] for m in metrics},
                                }
                            
                            for metric in metrics:
                                if metric in file_data:
                                    trace_data[qps][method][metric].append(file_data[metric])
    
    if available_qps:
        qps_list = sorted(list(available_qps))
    else:
        # Fallback to paper defaults if no data found
        qps_list = [7.00, 7.25, 7.50, 7.75, 8.00]
    
    print(f"Found data for QPS values: {qps_list}")
    print(f"Found data for methods: {list(available_methods)}")
    
    # Ensure every qps has all methods initialized
    for qps in qps_list:
        if qps not in trace_data:
            trace_data[qps] = {
                'Llumnix': {m: [] for m in metrics},
                'All': {m: [] for m in metrics},
                'All-Wait': {m: [] for m in metrics},
            }
    
    return qps_list, trace_data


def get_figure11_data(log_path, trace):
    qps_list, trace_data = get_trace_data(log_path, trace)
    # latency_dict: QPS->'Request'/'Prefill'/'Decode'/'Preemption'->[P99, Mean]
    llumnix_latency_dict = {}
    all_latency_dict = {}
    all_wait_latency_dict = {}
    
    for latency_dict in [llumnix_latency_dict, all_latency_dict, all_wait_latency_dict]:
        for qps in qps_list:
            latency_dict[qps] = {
                'Request': [0.0, 0.0], 
                'Prefill': [0.0, 0.0], 
                'Decode': [0.0, 0.0], 
                'Preemption': [0.0]
            }

    methods = ['Llumnix', 'All', 'All-Wait']
    method_dicts = {
        'Llumnix': llumnix_latency_dict,
        'All': all_latency_dict,
        'All-Wait': all_wait_latency_dict
    }

    for qps in qps_list:
        for method in methods:
            if method in trace_data[qps] and len(trace_data[qps][method]['Request P99']) != 0:
                data_dict = trace_data[qps][method]
                latency_dict = method_dicts[method]
                
                # Calculate averages for each metric
                if data_dict['Request P99']:
                    latency_dict[qps]['Request'][0] = sum(data_dict['Request P99']) / len(data_dict['Request P99'])
                if data_dict['Request Mean']:
                    latency_dict[qps]['Request'][1] = sum(data_dict['Request Mean']) / len(data_dict['Request Mean'])
                if data_dict['Prefill P99']:
                    latency_dict[qps]['Prefill'][0] = sum(data_dict['Prefill P99']) / len(data_dict['Prefill P99'])
                if data_dict['Prefill Mean']:
                    latency_dict[qps]['Prefill'][1] = sum(data_dict['Prefill Mean']) / len(data_dict['Prefill Mean'])
                if data_dict['Decode P99']:
                    latency_dict[qps]['Decode'][0] = sum(data_dict['Decode P99']) / len(data_dict['Decode P99'])
                if data_dict['Decode Mean']:
                    latency_dict[qps]['Decode'][1] = sum(data_dict['Decode Mean']) / len(data_dict['Decode Mean'])
                if data_dict['Preemption Loss']:
                    latency_dict[qps]['Preemption'][0] = sum(data_dict['Preemption Loss']) / len(data_dict['Preemption Loss'])

    return qps_list, llumnix_latency_dict, all_latency_dict, all_wait_latency_dict


def plot_one_trace(trace, axs, qps_list, llumnix_latency_dict, all_latency_dict, all_wait_latency_dict):
    def plot_one_metric(ax, metric_key, metric_index):
        # Get values for each method, handling cases where data might be missing
        llumnix_values = []
        all_values = []
        all_wait_values = []
        
        for qps in qps_list:
            # Llumnix values
            if qps in llumnix_latency_dict and llumnix_latency_dict[qps][metric_key][metric_index] > 0:
                llumnix_values.append(llumnix_latency_dict[qps][metric_key][metric_index])
            else:
                llumnix_values.append(None)
            
            # All values
            if qps in all_latency_dict and all_latency_dict[qps][metric_key][metric_index] > 0:
                all_values.append(all_latency_dict[qps][metric_key][metric_index])
            else:
                all_values.append(None)
            
            # All-Wait values
            if qps in all_wait_latency_dict and all_wait_latency_dict[qps][metric_key][metric_index] > 0:
                all_wait_values.append(all_wait_latency_dict[qps][metric_key][metric_index])
            else:
                all_wait_values.append(None)

        colors = get_colors()

        # Plot each method only if it has valid data
        if any(v is not None for v in llumnix_values):
            valid_qps = [qps for i, qps in enumerate(qps_list) if llumnix_values[i] is not None]
            valid_values = [v for v in llumnix_values if v is not None]
            if valid_values:
                ax.plot(valid_qps, valid_values, marker='s', linestyle='-', color=colors[1], label='Llumnix')
        
        if any(v is not None for v in all_values):
            valid_qps = [qps for i, qps in enumerate(qps_list) if all_values[i] is not None]
            valid_values = [v for v in all_values if v is not None]
            if valid_values:
                ax.plot(valid_qps, valid_values, marker='o', linestyle='--', color=colors[0], label='All')
        
        if any(v is not None for v in all_wait_values):
            valid_qps = [qps for i, qps in enumerate(qps_list) if all_wait_values[i] is not None]
            valid_values = [v for v in all_wait_values if v is not None]
            if valid_values:
                ax.plot(valid_qps, valid_values, marker='^', linestyle=':', color=colors[2], label='Seq-Wait')

        fontsize=15

        if metric_key == 'Request' and metric_index == 0:
            ax.set_ylabel('ShareGPT' + '\n\n' + 'Latency (s)', fontsize=fontsize)

        # Set y-axis limits only if we have data
        all_valid_values = [v for v in llumnix_values + all_values + all_wait_values if v is not None]
        if all_valid_values:
            ymax = max(all_valid_values)
            ax.set_ylim(0 - ymax * 0.1, ymax + ymax * 0.1)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        if trace == "ShareGPT":
            if metric_key == 'Request' and metric_index == 0:
                ax.set_title('Request P99', fontsize=fontsize)
            elif metric_key == 'Request' and metric_index == 1:
                ax.set_title('Request Mean', fontsize=fontsize)
            elif metric_key == 'Prefill' and metric_index == 0:
                ax.set_title('Prefill P99', fontsize=fontsize)
            elif metric_key == 'Prefill' and metric_index == 1:
                ax.set_title('Prefill Mean', fontsize=fontsize)
            elif metric_key == 'Decode' and metric_index == 0:
                ax.set_title('Decode P99', fontsize=fontsize)
            elif metric_key == 'Decode' and metric_index == 1:
                ax.set_title('Decode Mean', fontsize=fontsize)
            elif metric_key == 'Preemption':
                ax.set_title('Preemption Loss', fontsize=fontsize)
        ax.grid(True)

    metric_keys = ['Request', 'Prefill', 'Decode', 'Preemption']
    for i in range(3 * 2 + 1):
        metric_key = metric_keys[int(i / 2)]
        metric_index = i % 2
        plot_one_metric(axs[i], metric_key, metric_index)


def plot_sharegpt_only(log_path):
    traces = ["ShareGPT"]

    wspace = 0.35
    hspace = 0.3
    nrows = 1
    ncols = 7
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.0 * ncols + wspace * (ncols - 1), 2.2 * nrows + hspace * (nrows - 1)))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs_dict = {"ShareGPT": axs}

    for trace in traces:
        try:
            qps_list, llumnix_latency_dict, all_latency_dict, all_wait_latency_dict = get_figure11_data(log_path, trace)
            
            # Check if we have any data at all
            if not qps_list:
                print(f"Warning: No data found for trace {trace}")
                continue
                
            plot_one_trace(trace, axs_dict[trace], qps_list, llumnix_latency_dict, all_latency_dict, all_wait_latency_dict)
        except Exception as e:
            print(f"Error processing trace {trace}: {e}")
            continue

    # Collect all unique legend handles and labels
    all_handles = []
    all_labels = []
    for ax in fig.axes:
        try:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in all_labels:
                    all_handles.append(handle)
                    all_labels.append(label)
        except Exception as e:
            print(f"Warning: Could not get legend from axis: {e}")
            continue
    
    fontsize=15
    if all_handles:
        fig.legend(all_handles, all_labels, loc="lower center", ncol=min(3, len(all_handles)), bbox_to_anchor=(0.5, 0.905), fontsize=fontsize)
    else:
        print("Warning: No legend handles found")

    fig.savefig("figure11_sharegpt_only.png", bbox_inches="tight")
    print(f"Figure saved as figure11_sharegpt_only.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', default='../log', type=str)
    args = parser.parse_args()

    plot_sharegpt_only(args.log_path)
