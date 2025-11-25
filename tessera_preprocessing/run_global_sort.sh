#!/usr/bin/env bash
#SBATCH --job-name=sort_index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --output=logs/sort_index_%j.log
#SBATCH --error=logs/sort_index_%j.err

set -euo pipefail

# Define paths
INPUT_DIR="/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index"
OUTPUT_DIR="/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_shuffled"
TEMP_DIR="/scratch/groups/dlobell/psinghal/dask_temp"
SCRIPT_PATH="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_preprocessing/global_shuffle_index.py"
CONTAINER_PATH="/scratch/groups/dlobell/psinghal/sentineldownloader/dependencies/apptainer/tesseraenv.sif"

# Create log/temp directories if they don't exist
mkdir -p logs
mkdir -p "$TEMP_DIR"

echo "Starting Global Index Sort"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Temp: $TEMP_DIR"
echo "Date: $(date)"

# Run the python script inside Apptainer
# We use ulimit to prevent 'Too many open files' error with Dask
apptainer exec "$CONTAINER_PATH" /bin/bash -c "ulimit -n 65535 && python \"$SCRIPT_PATH\" \
    --input_dir \"$INPUT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --temp_dir \"$TEMP_DIR\" \
    --memory_limit 60GB"

echo "Job finished at $(date)"

