# If want to debug, ignore this file and tweak options in launch.json
export IP_PORTS=("172.31.9.153:8000") # Private IPs of servers. Don't use 6379, which will be used by ray
export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
export NUM_PROMPTS=$2
export QPS=$1
export DATASET_PATH=/mnt/data/huggingface_cache/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a/sharegpt_gpt4.jsonl
# export DATASET_PATH=/home/ubuntu/reshardLLM/datasets/BurstGPT/data/BurstGPT_1.csv # Need to change dataset_type
export OUTPUT=output.log

cd ../benchmark

python benchmark_serving.py \
    --ip_ports ${IP_PORTS[@]} \
    --tokenizer $MODEL_PATH \
    --random_prompt_count $NUM_PROMPTS \
    --dataset_type "sharegpt" \
    --dataset_path $DATASET_PATH \
    --qps $QPS \
    --distribution "poisson" \
    --log_latencies \
    --fail_on_response_failure \
    --output_file $OUTPUT

# If INSTANCE_CSV is set and the file exists, plot GPU utilization and Figure 12 (fragmentation proportion).
if [ -n "${INSTANCE_CSV:-}" ] && [ -f "$INSTANCE_CSV" ]; then
    python plot_instance_metrics.py --instance-csv "$INSTANCE_CSV" --output-dir .
fi