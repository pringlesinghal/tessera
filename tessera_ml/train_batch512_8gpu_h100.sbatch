#!/usr/bin/env bash
#SBATCH --job-name=train_batch512_8gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=120:00:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=8
#SBATCH -C GPU_SKU:H100_SXM5,GPU_MEM:80GB
#SBATCH --output=logs/main_training_8gpu_batch512_h100_%j.log
#SBATCH --error=logs/main_training_8gpu_batch512_h100_%j.err

set -euo pipefail

# Configuration
CONFIG_FILE_ARG="${1:-configs/ssl_config_test_batch2_modified.py}"
NUM_GPUS=8
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"

# Change to the tessera_ml directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

# Resolve config file path
if [[ "$CONFIG_FILE_ARG" = /* ]]; then
    CONFIG_FILE="$CONFIG_FILE_ARG"
else
    CONFIG_FILE="$WORK_DIR/$CONFIG_FILE_ARG"
fi

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $WORK_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "Container: $CONTAINER_IMAGE"
echo "========================================"

# Create logs directory
mkdir -p logs

# CRITICAL: NCCL configuration to fix socket errors
export NCCL_SOCKET_IFNAME="ib0,ib1,em1"   # Prioritize IB, fallback to ethernet
export NCCL_IB_DISABLE=0                   # Enable InfiniBand
export NCCL_P2P_DISABLE=0                  # Enable P2P transfers
export NCCL_DEBUG=WARN                     # Reduce verbosity
export NCCL_ASYNC_ERROR_HANDLING=1         # Better error handling

# Fix IPv6 socket issues
export NCCL_SOCKET_FAMILY=AF_INET          # Force IPv4 only
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # All 8 GPUs

# PyTorch distributed settings - fix communication issues
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO        # Detailed logging for debugging

# CPU performance optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print system information for debugging
echo "Network interfaces:"
ip addr show | grep -E '^[0-9]+:|inet ' | grep -v inet6 || echo "Network info unavailable"

echo "GPU information:"
nvidia-smi --list-gpus

echo "GPU topology:"
nvidia-smi topo -m || echo "GPU topology unavailable"

# Set bind paths for container
PARENT_DIR="$(dirname $WORK_DIR)"
BIND_PATHS="$WORK_DIR:$WORK_DIR,$PARENT_DIR:$PARENT_DIR,/scratch:/scratch"

echo "Bind paths: $BIND_PATHS"

# Set PYTHONPATH for imports
export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

echo "========================================"
echo "Starting 8-GPU training with batch size 512 on H100..."
echo "Config: ssl_config_test_batch2_modified.py"
echo "Target: batch_size=512 per GPU (4096 global batch)"
echo "========================================"

# Run training with fixed networking
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
    --env NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
    --env NCCL_DEBUG="$NCCL_DEBUG" \
    --env NCCL_SOCKET_FAMILY="$NCCL_SOCKET_FAMILY" \
    --env NCCL_ASYNC_ERROR_HANDLING="$NCCL_ASYNC_ERROR_HANDLING" \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING="$TORCH_NCCL_ASYNC_ERROR_HANDLING" \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint="127.0.0.1:29500" \
        "$WORK_DIR/train_ssl_multi_gpu.py" \
        --config "$CONFIG_FILE"

echo "========================================"
echo "Training completed!"
echo "========================================"
