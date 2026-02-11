# $1 for head/worker ID (0 is head, 1 is worker 1, etc.)
# $2 for number of initial instances
# $3 for GPU memory utilization allowed for KV cache (1.0 means 100% of GPU memory)
# Change HOST, whether migration in configs/vllm_*.yml

export HEAD_NODE_IP='172.31.9.153'
export HOST="0.0.0.0"
export PORT=$2
export GPU_UTILIZATION=$3
export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
# export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5

# Determine if this is head node or worker node
if test $1 -eq 0; then
    export IS_HEAD_NODE=1
    echo "Creating head node. Listening on $HOST:$PORT. Head node ip: $HEAD_NODE_IP. Init instances: 1"
    export CONFIG_PATH=/home/ubuntu/reshardLLM/reshardLLM/configs/vllm_head.yml
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
    --max-model-len 2000 \
    --head-node $IS_HEAD_NODE \
    --head-node-ip $HEAD_NODE_IP

