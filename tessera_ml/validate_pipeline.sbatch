#!/usr/bin/env bash
#SBATCH --job-name=validate_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=1
#SBATCH -C GPU_SKU:A100_SXM4,GPU_MEM:80GB
#SBATCH --output=logs/validate_pipeline_%j.log
#SBATCH --error=logs/validate_pipeline_%j.err

set -euo pipefail

# Configuration
CHECKPOINT_PATH="${1:-/scratch/groups/dlobell/psinghal/sentineldownloader/tessera_best_model_fsdp_20250427_084307.pt}"
N_SAMPLES="${2:-100}"
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"

# Change to the tessera_ml directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

echo "========================================"
echo "SSL Data Curation Pipeline Validation"
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "N samples: $N_SAMPLES"
echo "========================================"

# Create logs directory
mkdir -p logs

# Set bind paths for container
PARENT_DIR="$(dirname $WORK_DIR)"
BIND_PATHS="$WORK_DIR:$WORK_DIR,$PARENT_DIR:$PARENT_DIR,/scratch:/scratch"

# Set PYTHONPATH for imports (ssl_data_curation is accessed as a subpackage of tessera_ml)
export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "PYTHONPATH: $PYTHONPATH"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run validation
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$CONTAINER_IMAGE" \
    python "$WORK_DIR/scripts/validate_pipeline.py" \
        --checkpoint "$CHECKPOINT_PATH" \
        --n_samples "$N_SAMPLES" \
        --device cuda

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "VALIDATION PASSED!"
    echo "========================================"
    echo ""
    echo "You can now run the full pipeline:"
    echo "  sbatch extract_embeddings.sbatch"
    echo "  sbatch --dependency=afterok:<JOB_ID> run_clustering.sbatch"
    echo "  sbatch --dependency=afterok:<JOB_ID> sample_curated_index.sbatch <target_size>"
else
    echo ""
    echo "========================================"
    echo "VALIDATION FAILED - Check logs above"
    echo "========================================"
fi

exit $EXIT_CODE
