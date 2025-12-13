#!/usr/bin/env bash
#SBATCH --job-name=fsdp_train_multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=8
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --output=logs/train_fsdp_multinode_%j.log
#SBATCH --error=logs/train_fsdp_multinode_%j.err

set -euo pipefail

# Configuration
CONFIG_FILE_ARG="${1:-configs/ssl_config.py}"
GPUS_PER_NODE=8
NUM_NODES=$SLURM_NNODES
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
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
echo "Nodes: $SLURM_NODELIST ($NUM_NODES nodes)"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Working directory: $WORK_DIR"
echo "Config file: $CONFIG_FILE"
echo "Container: $CONTAINER_IMAGE"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set NCCL environment variables for multi-node training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# Set OMP threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Get master node address for distributed training
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Print GPU information on master node
echo "Available GPUs on master node:"
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
echo "Starting multi-node training with torchrun in Apptainer..."
echo "========================================"

# Set PYTHONPATH to include parent directory (for tessera_ml imports) and current directory (for local imports)
PARENT_DIR="$(dirname "$WORK_DIR")"
export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Run training with torchrun (multi-node) inside container
srun apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        "$WORK_DIR/train_ssl_multi_gpu.py" \
        --config "$CONFIG_FILE"

echo "========================================"
echo "Multi-node training completed!"
echo "========================================"

