#!/usr/bin/env bash
#SBATCH --job-name=ssl_train_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=1
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --output=logs/train_1gpu_%j.log
#SBATCH --error=logs/train_1gpu_%j.err

set -euo pipefail

# Configuration
CONFIG_FILE_ARG="${1:-configs/ssl_config.py}"
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
echo "Config file: $CONFIG_FILE"
echo "Container: $CONTAINER_IMAGE"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

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
BIND_PATHS="$BIND_PATHS,/scratch:/scratch"

# Add additional bind paths if your data is elsewhere (uncomment and modify as needed)
# BIND_PATHS="$BIND_PATHS,/path/to/your/data:/data"

echo "Bind paths: $BIND_PATHS"

echo "========================================"
echo "Starting single-GPU training with Apptainer..."
echo "========================================"

# Run training with single GPU inside container
# Set PYTHONPATH to include parent directory (for tessera_ml imports) and current directory (for local imports)
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$(dirname "$WORK_DIR"):$WORK_DIR:$PYTHONPATH" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=1 \
        "$WORK_DIR/train_ssl_multi_gpu.py" \
        --config "$CONFIG_FILE"

echo "========================================"
echo "Training completed!"
echo "========================================"

