#!/usr/bin/env bash
#SBATCH --job-name=ssl_clustering
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=1
#SBATCH -C GPU_SKU:A100_SXM4,GPU_MEM:80GB
#SBATCH --output=logs/run_clustering_%j.log
#SBATCH --error=logs/run_clustering_%j.err

set -euo pipefail

# Configuration
EMBEDDINGS_PATH="${1:-curation_outputs/embeddings.npy}"
OUTPUT_DIR="${2:-curation_outputs}"
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"

# Clustering parameters (can be overridden via environment variables)
# 4-level hierarchical clustering for 50M samples
# Based on Meta FAIR SSL data curation paper (arXiv:2405.15613)
# Level 1: 50M/100K = 500 avg cluster size
# Level 2: 100K/5K = 20 avg cluster size
# Level 3: 5K/500 = 10 avg cluster size
# Level 4: 500/100 = 5 avg cluster size
N_LEVELS="${N_LEVELS:-4}"
N_CLUSTERS="${N_CLUSTERS:-100000 5000 500 100}"
SAMPLE_SIZES="${SAMPLE_SIZES:-1 10 5 3}"  # ~half avg cluster size per level
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-closest}"

# Change to the tessera_ml directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $WORK_DIR"
echo "Embeddings path: $EMBEDDINGS_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "N_LEVELS: $N_LEVELS"
echo "N_CLUSTERS: $N_CLUSTERS"
echo "Container: $CONTAINER_IMAGE"
echo "========================================"

# Create logs directory
mkdir -p logs

# Set bind paths for container
PARENT_DIR="$(dirname $WORK_DIR)"
BIND_PATHS="$WORK_DIR:$WORK_DIR,$PARENT_DIR:$PARENT_DIR,/scratch:/scratch"

# Set PYTHONPATH for imports (ssl_data_curation is accessed as a subpackage of tessera_ml)
export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Bind paths: $BIND_PATHS"
echo "PYTHONPATH: $PYTHONPATH"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CPU performance optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "========================================"
echo "Starting hierarchical k-means clustering..."
echo "========================================"

# Run clustering script
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    "$CONTAINER_IMAGE" \
    python "$WORK_DIR/scripts/run_clustering.py" \
        --embeddings_path "$EMBEDDINGS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --n_levels $N_LEVELS \
        --n_clusters $N_CLUSTERS \
        --sample_sizes $SAMPLE_SIZES \
        --sample_strategy "$SAMPLE_STRATEGY" \
        --device cuda

echo "========================================"
echo "Clustering completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
