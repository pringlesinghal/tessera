#!/usr/bin/env bash
#SBATCH --job-name=extract_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=8
#SBATCH -C GPU_SKU:A100_SXM4,GPU_MEM:80GB
#SBATCH --output=logs/extract_embeddings_%j.log
#SBATCH --error=logs/extract_embeddings_%j.err

set -euo pipefail

# Configuration
CHECKPOINT_PATH="${1:-/scratch/groups/dlobell/psinghal/sentineldownloader/tessera_best_model_fsdp_20250427_084307.pt}"
OUTPUT_DIR="${2:-curation_outputs}"
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"
NUM_GPUS=8

# Change to the tessera_ml directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $WORK_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Container: $CONTAINER_IMAGE"
echo "========================================"

# Create output directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Set bind paths for container
PARENT_DIR="$(dirname $WORK_DIR)"
BIND_PATHS="$WORK_DIR:$WORK_DIR,$PARENT_DIR:$PARENT_DIR,/scratch:/scratch"

# Set PYTHONPATH for imports
export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Bind paths: $BIND_PATHS"
echo "PYTHONPATH: $PYTHONPATH"

# NCCL environment variables for single-node multi-GPU (matches training script)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_NET="Socket"
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CPU performance optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "========================================"
echo "Starting embedding extraction with torchrun ($NUM_GPUS GPUs)..."
echo "========================================"

# Check if this is a resumed job (per-rank checkpoints)
RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/extraction_checkpoint_rank0.json" ]; then
    echo "Found per-rank checkpoint files, enabling resume mode"
    RESUME_FLAG="--resume"
fi

# Run extraction with torchrun (one process per GPU)
# Each process gets: batch_size=512, num_workers=2, cache_size=100
# Total effective: batch_size=4096, workers=16, cache=800
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    --env NCCL_DEBUG="$NCCL_DEBUG" \
    --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
    --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
    --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    --env NCCL_IGNORE_DISABLED_P2P="$NCCL_IGNORE_DISABLED_P2P" \
    --env NCCL_NET="$NCCL_NET" \
    --env NCCL_NSOCKS_PERTHREAD="$NCCL_NSOCKS_PERTHREAD" \
    --env NCCL_SOCKET_NTHREADS="$NCCL_SOCKET_NTHREADS" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        "$WORK_DIR/scripts/extract_embeddings.py" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 512 \
        --num_workers 2 \
        --checkpoint_interval 5000 \
        --max_samples 50000000 \
        $RESUME_FLAG

echo "========================================"
echo "Embedding extraction completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
