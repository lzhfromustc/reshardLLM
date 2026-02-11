import argparse
import os
import json
import numpy as np
import re
import pandas as pd
from collections import defaultdict
from vllm.simulator.profiling import *


def _pad_to_alignment(x, multiple_of):
    return x + ((-1*x) % multiple_of)


def get_req_data(json_filename, key):
    with open(json_filename, 'r') as f:
        latency_info = json.load(f)[0]
    req_data = latency_info[key]
    if key in ['prefill_latencies', 'decode_latencies']:
        req_data = [i / 1000 for i in req_data]

    return req_data


def get_mean_and_p99_latency(latencies):
    latencies = list(latencies)
    mean_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    return mean_latency, p99_latency


def parse_dir_name(log_filename, conversation_mode="first"):
    dir_path = os.path.dirname(log_filename)
    dir_name = os.path.basename(dir_path)

    qps_match = re.search(r"qps(\d+(\.\d+)?)", dir_name)
    imean_match = re.search(r"imean(\d+)", dir_name)
    omean_match = re.search(r"omean(\d+)", dir_name)
    migrate_match = re.search(r"migrate(\d+)", dir_name)
    defrag_match = re.search(r"defrag(\d+)", dir_name)

    qps = float(qps_match.group(1)) if qps_match else None
    imean = int(imean_match.group(1)) if imean_match else None
    omean = int(omean_match.group(1)) if omean_match else None
    migrate = int(migrate_match.group(1)) if migrate_match else None
    defrag = int(defrag_match.group(1)) if defrag_match else None

    trace = str(imean) + '_' + str(omean)

    # Method based on conversation mode
    if conversation_mode == "first":
                method = 'Llumnix'
    elif conversation_mode == "all":
        method = 'All'
    elif conversation_mode == "all-wait":
        method = 'All-Wait'
    else:
        method = 'Llumnix'  # Default fallback

    return qps, trace, method


def get_preemption_loss(log_filename):
    try:
    df = pd.read_csv(log_filename + "_req.csv").drop_duplicates()
    df = df.sort_values(by='timestamp')
    preempted_request_set = set()
    request_num = len(df["req_id"].drop_duplicates())
    preemption_loss_sum = 0
    last_killed_time = defaultdict(lambda: 0.0)
    last_killed_len = defaultdict(lambda: 0.0)

        # Try to use the profiling database, but fall back to a simpler calculation if it's not available
        try:
    database = ProfilingDatabase("/mnt/huangziming/artifact/vllm_017/vllm/vllm/simulator/profiling_result_new.pkl", False)
    profiling_result = database.get("llama-7b")
    sim_parallel_config = SimParallelConfig(1, 1)
    latency_mem = profiling_result.para_dict[sim_parallel_config]

    for _, row in df.iterrows():
        req_id = row["req_id"]
        if row["event"] == "prefill" and last_killed_time[req_id]:
            preemption_loss_sum += row["timestamp"] - last_killed_time[req_id]
            prompt_len = last_killed_len[req_id]
            prompt_len = _pad_to_alignment(prompt_len, 8)
            preemption_loss_sum += (latency_mem.prefill_latency[(1, prompt_len)][0] - latency_mem.decode_latency[(8, last_killed_len[req_id])][0]) / 1000
            preempted_request_set.add(req_id)
        elif row["event"] == "killed":
            last_killed_time[req_id] = row["timestamp"]
            last_killed_len[req_id] = row["output_len"]
            
            preemption_loss = preemption_loss_sum / request_num if request_num > 0 else 0
    return preemption_loss

        except (FileNotFoundError, ImportError, Exception) as e:
            print(f"Warning: Profiling database not available ({e}), using simplified preemption loss calculation")
            
            # Fallback: simple preemption loss based on killed requests
            killed_requests = df[df['event'] == 'killed']
            if len(killed_requests) > 0:
                # Estimate preemption loss as percentage of killed requests
                preemption_loss = len(killed_requests) / request_num if request_num > 0 else 0
                return preemption_loss
            else:
                return 0.0
                
    except Exception as e:
        print(f"Error in get_preemption_loss: {e}")
        return 0.0


def get_evaluation_data(log_filename, conversation_mode="first"):
    # dir, json
    # trace_key, method(llumnix, infaas++, round-robin), qps
    # request/prefill/decode mean/p99, preemption loss

    # read dir
    qps, trace, method = parse_dir_name(log_filename, conversation_mode)

    data_file.write("Trace: {}\n".format(trace))
    data_file.write("Method: {}\n".format(method))
    data_file.write("QPS: {:.2f}\n".format(qps))
    
    json_filename = os.path.splitext(log_filename)[0] + "_latency_info.json"
    
    # Check if latency info file exists
    if not os.path.exists(json_filename):
        data_file.write("Error: Latency info file not found: {}\n".format(json_filename))
        return
    
    # Load full JSON once to get conversation-level array if present
    try:
        with open(json_filename, 'r') as jf:
            full_info = json.load(jf)
    except Exception as e:
        data_file.write("Error reading JSON: {}\n".format(e))
        return

    # If all-wait and conversation_latencies present, compute Request metrics from them
    conv_latencies = []
    if conversation_mode == 'all-wait':
        try:
            if full_info and isinstance(full_info, list) and 'conversation_latencies' in full_info[0]:
                conv_latencies = list(full_info[0]['conversation_latencies'])
        except Exception:
            conv_latencies = []

    data_keys = ['request_latencies', 'prefill_latencies', 'decode_latencies']
    key2metric = {'request_latencies': 'Request', 'prefill_latencies': 'Prefill', 'decode_latencies': 'Decode'}
    
    for data_key in data_keys:
        if data_key == 'request_latencies' and conv_latencies:
            # Use conversation-level latencies for Request metrics
            mean_latency, p99_latency = get_mean_and_p99_latency(conv_latencies)
        else:
        latencies = get_req_data(json_filename, data_key)
            if not latencies:
                data_file.write("{}: No data available\n".format(key2metric[data_key]))
                continue
        mean_latency, p99_latency = get_mean_and_p99_latency(latencies)
        mean_metric_name = key2metric[data_key] + ' ' + 'Mean'
        p99_metric_name = key2metric[data_key] + ' ' + 'P99'
        data_file.write("{}: {:.4f}\n".format(p99_metric_name, p99_latency))
        data_file.write("{}: {:.4f}\n".format(mean_metric_name, mean_latency))

    try:
        # Check if required CSV files exist for preemption loss calculation
        req_csv_filename = log_filename + "_req.csv"
        if os.path.exists(req_csv_filename):
    preemption_loss = get_preemption_loss(log_filename)
    data_file.write("Preemption Loss: {:.4f}".format(preemption_loss))
        else:
            data_file.write("Preemption Loss: No request CSV data available")
    except Exception as e:
        data_file.write("Preemption Loss: Error calculating - {}\n".format(str(e)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-filename', type=str, required=True,
                        help='Path to the log file to process')
    parser.add_argument('--conversation-mode', type=str, choices=['first', 'all', 'all-wait'], default='first',
                        help='Conversation mode used in the benchmark (first/all/all-wait)')
    args = parser.parse_args()

    prefix, _ = os.path.splitext(args.log_filename)
    data_filename = prefix + '.data'
    data_file = open(data_filename, 'w')

    # Write conversation mode info to data file
    data_file.write("Conversation Mode: {}\n".format(args.conversation_mode))
    data_file.write("Processing Time: {}\n".format(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")))
    data_file.write("-" * 50 + "\n")
    
    print(f"Processing log with conversation mode: {args.conversation_mode}")

    try:
        get_evaluation_data(args.log_filename, args.conversation_mode)
    except Exception as e:
        print(e)
        data_file.write("Invalid Log!\n")
    finally:
        data_file.close()
