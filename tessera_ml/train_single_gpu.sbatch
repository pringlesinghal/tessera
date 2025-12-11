#!/usr/bin/env bash
#SBATCH --job-name=ssl_train_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --output=logs/train_1gpu_%j.log
#SBATCH --error=logs/train_1gpu_%j.err

set -euo pipefail

# Configuration
CONFIG_FILE="${1:-configs/ssl_config.py}"
CONTAINER_IMAGE="../ml_env.sif"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config file: $CONFIG_FILE"
echo "Container: $CONTAINER_IMAGE"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to the tessera_ml directory
cd "$(dirname "$0")"
WORK_DIR="$(pwd)"
echo "Working directory: $WORK_DIR"

# Set OMP threads for better CPU performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Set bind paths for container (adjust these for your data locations)
# Bind current directory, parent directory (for tessera_ml.dataset), and common data paths
BIND_PATHS="$WORK_DIR:$WORK_DIR"
BIND_PATHS="$BIND_PATHS,$(dirname $WORK_DIR):$(dirname $WORK_DIR)"

# Add additional bind paths if your data is elsewhere (uncomment and modify as needed)
# BIND_PATHS="$BIND_PATHS,/scratch:/scratch"
# BIND_PATHS="$BIND_PATHS,/path/to/your/data:/data"

echo "Bind paths: $BIND_PATHS"

echo "========================================"
echo "Starting single-GPU training with Apptainer..."
echo "========================================"

# Run training with single GPU inside container
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=1 \
        train_ssl_multi_gpu.py \
        --config "$CONFIG_FILE"

echo "========================================"
echo "Training completed!"
echo "========================================"

