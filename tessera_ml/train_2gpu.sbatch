#!/usr/bin/env bash
#SBATCH --job-name=fsdp_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=2
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --output=logs/train_2gpu_%j.log
#SBATCH --error=logs/train_2gpu_%j.err

set -euo pipefail

# Configuration
CONFIG_FILE_ARG="${1:-configs/ssl_config_2gpu.py}"  # Default: 2 GPU test config
NUM_GPUS=2
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"

# Change to the tessera_ml directory (explicit path)
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

# Resolve config file path (make it absolute if it's relative)
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Set NCCL environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_SOCKET_IFNAME=^docker0,lo

# Set OMP threads for better CPU performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# CUDA memory allocator config to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print GPU information
echo "Available GPUs:"
nvidia-smi --list-gpus

# Set bind paths for container (adjust these for your data locations)
# Bind current directory, parent directory (for tessera_ml.dataset), and common data paths
BIND_PATHS="$WORK_DIR:$WORK_DIR"
BIND_PATHS="$BIND_PATHS,$(dirname $WORK_DIR):$(dirname $WORK_DIR)"
BIND_PATHS="$BIND_PATHS,/scratch:/scratch"

# Add additional bind paths if your data is elsewhere (uncomment and modify as needed)
# BIND_PATHS="$BIND_PATHS,/path/to/your/data:/data"

echo "Bind paths: $BIND_PATHS"

echo "========================================"
echo "Starting training with torchrun in Apptainer..."
echo "========================================"

# Set PYTHONPATH to include parent directory (for tessera_ml imports) and current directory (for local imports)
PARENT_DIR="$(dirname "$WORK_DIR")"
export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Run training with torchrun inside container
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        "$WORK_DIR/train_ssl_multi_gpu.py" \
        --config "$CONFIG_FILE"

echo "========================================"
echo "Training completed!"
echo "========================================"

