#!/bin/bash
# Script to start 4 TP2 instances on an 8-GPU machine
# Usage: bash start_4_tp2_instances.sh <GPU_UTILIZATION>

export HEAD_NODE_IP='172.31.9.153'
export HOST="0.0.0.0"
export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
# export MODEL_PATH=/mnt/data/huggingface_cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5
export GPU_UTILIZATION=${1:-0.9}
export TENSOR_PARALLEL_SIZE=2
export INITIAL_INSTANCES=1

echo "Starting 4 TP2 instances on an 8-GPU machine..."
echo "GPU utilization: $GPU_UTILIZATION"
echo ""
echo "Instance configuration:"
echo "  Instance 0: GPUs 0,1   Port 8000"
echo "  Instance 1: GPUs 2,3   Port 8001"
echo "  Instance 2: GPUs 4,5   Port 8002"
echo "  Instance 3: GPUs 6,7   Port 8003"
echo ""

export CONFIG_PATH=/home/ubuntu/reshardLLM/reshardLLM/configs/vllm_head.yml

# Start all 4 instances in background
for INSTANCE_ID in 0 1 2 3; do
    export PORT=$((8000 + INSTANCE_ID))
    export GPU_START=$((INSTANCE_ID * 2))
    export CUDA_VISIBLE_DEVICES="$GPU_START,$((GPU_START + 1))"
    
    echo "Starting instance $INSTANCE_ID (GPUs $CUDA_VISIBLE_DEVICES, Port $PORT)..."
    
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m llumnix.entrypoints.vllm.api_server \
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
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --head-node 1 \
        --head-node-ip $HEAD_NODE_IP &
    
    echo "  Instance $INSTANCE_ID started"
    echo ""
done

echo "All 4 TP2 instances are starting in background..."
echo "Waiting for initialization (about 30 seconds)..."

