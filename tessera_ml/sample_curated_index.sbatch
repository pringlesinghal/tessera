#!/usr/bin/env bash
#SBATCH --job-name=sample_curated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=serc
#SBATCH --output=logs/sample_curated_%j.log
#SBATCH --error=logs/sample_curated_%j.err

set -euo pipefail

# Configuration
TARGET_SIZE="${1:-100000}"
CLUSTER_DIR="${2:-curation_outputs}"
OUTPUT_DIR="${3:-curation_outputs/curated_index_${TARGET_SIZE}}"
STRATEGY="${4:-r}"
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"

# Change to the tessera_ml directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $WORK_DIR"
echo "Target size: $TARGET_SIZE"
echo "Cluster directory: $CLUSTER_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Strategy: $STRATEGY"
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

echo "========================================"
echo "Starting curated index sampling..."
echo "========================================"

# Run sampling script
apptainer exec \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$CONTAINER_IMAGE" \
    python "$WORK_DIR/scripts/sample_curated_index.py" \
        --cluster_dir "$CLUSTER_DIR" \
        --index_mapping_path "$CLUSTER_DIR/index_mapping.npy" \
        --original_index_dir "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid" \
        --output_dir "$OUTPUT_DIR" \
        --target_size "$TARGET_SIZE" \
        --strategy "$STRATEGY"

echo "========================================"
echo "Curated index creation completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "To use this curated index for training, update your config:"
echo "  config['index_dir'] = '$OUTPUT_DIR'"
echo ""
echo "Or run training with:"
echo "  sbatch train_batch512_8gpu.sbatch configs/ssl_config_curated.py"
