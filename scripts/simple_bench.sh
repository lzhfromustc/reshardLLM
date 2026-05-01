# $1 head/worker ID (0 = head, 1+ = worker; plain vLLM only supports 0)
# $2 port
# $3 GPU memory utilization for KV cache (1.0 = 100%)
# $4 optional backend: llumnix (default) | vllm
# Change HOST, migration, etc. in configs/vllm_*.yml for Llumnix.

export HEAD_NODE_IP='172.31.9.153'
export HOST="0.0.0.0"
export PORT=$2
export GPU_UTILIZATION=$3
BACKEND="${4:-llumnix}"
# export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5

MAX_MODEL_LEN=16384

if [ "$BACKEND" = "vllm" ]; then
    if [ "$1" -ne 0 ]; then
        echo "Error: vLLM backend only runs a single server; use head id 0 (got $1)."
        exit 1
    fi
    echo "Starting vanilla vLLM (OpenAI API) on $HOST:$PORT (benchmarks that use /generate_benchmark need Llumnix)."
    exec python -m vllm.entrypoints.openai.api_server \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTILIZATION"
else
    echo "Starting llumnix on $HOST:$PORT"
fi

if [ "$BACKEND" != "llumnix" ]; then
    echo "Error: unknown backend '$BACKEND' (use llumnix or vllm)."
    exit 1
fi

# Determine if this is head node or worker node
if test $1 -eq 0; then
    export IS_HEAD_NODE=1
    echo "Creating head node. Listening on $HOST:$PORT. Head node ip: $HEAD_NODE_IP. Init instances: 1"
    export CONFIG_PATH=/home/ubuntu/reshardLLM/reshardLLM/configs/vllm_head.yml
    echo "With vllm_head.yml, instance CSV (LOG_INSTANCE_INFO: True) is written as server.log_instance.csv in this directory (head node)."
else
    export IS_HEAD_NODE=0
    echo "Creating worker node. Listening on $HOST:$PORT. Head node ip: $HEAD_NODE_IP. Init instances: 1"
    export CONFIG_PATH=/home/ubuntu/reshardLLM/reshardLLM/configs/vllm_worker$1.yml
fi

# Launch the Llumnix API server
python -m llumnix.entrypoints.vllm.api_server \
    --config-file $CONFIG_PATH \
    --initial-instances 1 \
    --launch-ray-cluster \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --worker-use-ray \
    --migration-backend rayrpc \
    --gpu-memory-utilization $GPU_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --head-node $IS_HEAD_NODE \
    --head-node-ip $HEAD_NODE_IP

