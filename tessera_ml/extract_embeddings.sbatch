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

# Change to the tessera_ml directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $WORK_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
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

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CPU performance optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "========================================"
echo "Starting embedding extraction..."
echo "========================================"

# Check if this is a resumed job (checkpoint exists)
RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/extraction_checkpoint.json" ]; then
    echo "Found checkpoint file, enabling resume mode"
    RESUME_FLAG="--resume"
fi

# Run extraction script
# batch_size 1024 is per-GPU, effective batch = 1024 * 8 = 8192
# num_workers 16 for good CPU utilization with 32 CPUs
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    "$CONTAINER_IMAGE" \
    python "$WORK_DIR/scripts/extract_embeddings.py" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 512 \
        --num_workers 16 \
        --device cuda \
        --checkpoint_interval 5000 \
        --max_samples 50000000 \
        $RESUME_FLAG

echo "========================================"
echo "Embedding extraction completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
