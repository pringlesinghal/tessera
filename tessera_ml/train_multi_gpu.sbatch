#!/usr/bin/env bash
#SBATCH --job-name=fsdp_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=8
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --output=logs/train_fsdp_%j.log
#SBATCH --error=logs/train_fsdp_%j.err

set -euo pipefail

# Configuration
CONFIG_FILE="${1:-configs/ssl_config.py}"
NUM_GPUS=8
CONTAINER_IMAGE="../ml_env.sif"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "Container: $CONTAINER_IMAGE"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to the tessera_ml directory
cd "$(dirname "$0")"
WORK_DIR="$(pwd)"
echo "Working directory: $WORK_DIR"

# Set NCCL environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_SOCKET_IFNAME=^docker0,lo

# Set OMP threads for better CPU performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Print GPU information
echo "Available GPUs:"
nvidia-smi --list-gpus

# Set bind paths for container (adjust these for your data locations)
# Bind current directory, parent directory (for tessera_ml.dataset), and common data paths
BIND_PATHS="$WORK_DIR:$WORK_DIR"
BIND_PATHS="$BIND_PATHS,$(dirname $WORK_DIR):$(dirname $WORK_DIR)"

# Add additional bind paths if your data is elsewhere (uncomment and modify as needed)
# BIND_PATHS="$BIND_PATHS,/scratch:/scratch"
# BIND_PATHS="$BIND_PATHS,/path/to/your/data:/data"

echo "Bind paths: $BIND_PATHS"

echo "========================================"
echo "Starting training with torchrun in Apptainer..."
echo "========================================"

# Run training with torchrun inside container
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train_ssl_multi_gpu.py \
        --config "$CONFIG_FILE"

echo "========================================"
echo "Training completed!"
echo "========================================"

