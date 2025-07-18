# $1 for head/worker ID (0 is head, 1 is worker 1, etc.)
# $2 for number of initial instances
# $3 for GPU memory utilization allowed for KV cache (1.0 means 100% of GPU memory)
# Change HOST, PORT, whether migration in configs/vllm_*.yml

export HEAD_NODE_IP='172.31.9.153'
export HOST="0.0.0.0"
export PORT=$2
export INITIAL_INSTANCES=1
export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
# export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5
export GPU_UTILIZATION=$3

if test $1 -eq 0; then
    echo "Creating head node. Listening on $HOST:$PORT. Head node ip: $HEAD_NODE_IP. Init instances:$INITIAL_INSTANCES."
    export CONFIG_PATH=/home/ubuntu/reshardLLM/reshardLLM/configs/vllm_head.yml
    python -m llumnix.entrypoints.vllm.api_server \
                --config-file $CONFIG_PATH \
                --initial-instances $INITIAL_INSTANCES \
                --launch-ray-cluster \
                --host $HOST \
                --port $PORT \
                --model $MODEL_PATH \
                --worker-use-ray \
                --migration-backend rayrpc \
                --gpu-memory-utilization $GPU_UTILIZATION \
                --max-model-len 2000 \
                --head-node 1 \
                --head-node-ip $HEAD_NODE_IP
else
    echo "Creating worker node with vllm_worker$1.yml. Listening on $HOST:$PORT. Head node ip: $HEAD_NODE_IP. Init instances:$INITIAL_INSTANCES."
    export CONFIG_PATH=/home/ubuntu/reshardLLM/reshardLLM/configs/vllm_worker$1.yml
    python -m llumnix.entrypoints.vllm.api_server \
            --config-file $CONFIG_PATH \
            --initial-instances $INITIAL_INSTANCES \
            --launch-ray-cluster \
            --host $HOST \
            --port $PORT \
            --model $MODEL_PATH \
            --worker-use-ray \
            --migration-backend rayrpc \
            --gpu-memory-utilization $GPU_UTILIZATION \
            --max-model-len 2000 \
            --head-node 0 \
            --head-node-ip $HEAD_NODE_IP
fi

