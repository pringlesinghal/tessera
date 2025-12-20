#!/usr/bin/env bash
#SBATCH --job-name=validation_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --output=logs/validation_dl_%A_%a_%j.log
#SBATCH --error=logs/validation_dl_%A_%a_%j.err
#SBATCH --array=0-84

set -euo pipefail
# Validation dataset download script
# Downloads tiles required for labeled tree species validation dataset

# File containing the tile IDs for validation dataset
TILE_LIST="${1:-/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/labelled_tree_data/required_tiles.txt}"
STARTED_FILE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/labelled_tree_data/validation_started_tiles.txt"
COMPLETED_FILE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/labelled_tree_data/validation_completed_tiles.txt"
FAILED_FILE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/labelled_tree_data/validation_failed_tiles.txt"

# Change to preprocessing directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_preprocessing

# Get the tile ID for this array task (line number based on array index)
TILE_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TILE_LIST")
echo "Processing validation tile: $TILE_ID (array task $SLURM_ARRAY_TASK_ID)"

# Create log directories if they don't exist
mkdir -p "$(dirname "$STARTED_FILE")/logs"

echo "$TILE_ID" >> "$STARTED_FILE"
echo "Saved started tile $TILE_ID to $STARTED_FILE"

# Track download success
DOWNLOAD_SUCCESS=true

# Loop over validation years: 2019-2023 (focus on recent years for validation)
for YEAR in {2024..2016..-1}; do
    echo "Processing year: $YEAR for validation tile $TILE_ID"
    
    # Check if tile already exists and is complete
    TILE_DIR="/scratch/groups/dlobell/shared_data/${YEAR}/${TILE_ID}"
    if [[ -d "$TILE_DIR/data_processed" ]]; then
        # Check for required files
        if [[ -f "$TILE_DIR/data_processed/bands.npy" && \
              -f "$TILE_DIR/data_processed/masks.npy" && \
              -f "$TILE_DIR/data_processed/doys.npy" ]]; then
            echo "Tile $TILE_ID year $YEAR already complete, skipping"
            continue
        else
            echo "Tile $TILE_ID year $YEAR incomplete, reprocessing"
            rm -rf "$TILE_DIR"
        fi
    fi
    
    # Download and process
    if bash s1_s2_downloader.sh "$TILE_ID" "$YEAR" && bash s1_s2_stacker.sh "$TILE_ID" "$YEAR"; then
        echo "Successfully processed $TILE_ID for year $YEAR"
        
        # Clean up raw data to save space
        RAW_DIR="/scratch/users/psinghal/time_series/${YEAR}/${TILE_ID}/data_raw"
        RAW_DIR_SAR="/scratch/users/psinghal/time_series/${YEAR}/${TILE_ID}/data_sar_raw" 
        LOG_DIR="/scratch/users/psinghal/time_series/${YEAR}/${TILE_ID}/logs"
        
        if [ -d "$RAW_DIR" ]; then
            rm -rf "$RAW_DIR"
            echo "Cleaned up raw directory $RAW_DIR"
        fi
        if [ -d "$RAW_DIR_SAR" ]; then
            rm -rf "$RAW_DIR_SAR"
            echo "Cleaned up raw SAR directory $RAW_DIR_SAR"
        fi
        if [ -d "$LOG_DIR" ]; then
            rm -rf "$LOG_DIR"
            echo "Cleaned up log directory $LOG_DIR"
        fi
    else
        echo "❌ Failed to process $TILE_ID for year $YEAR"
        DOWNLOAD_SUCCESS=false
    fi
done

# Record completion status
if [ "$DOWNLOAD_SUCCESS" = true ]; then
    echo "✅ Finished validation tile: $TILE_ID"
    echo "$TILE_ID" >> "$COMPLETED_FILE"
    echo "Saved completed tile $TILE_ID to $COMPLETED_FILE"
else
    echo "⚠️ Partial failure for validation tile: $TILE_ID"
    echo "$TILE_ID" >> "$FAILED_FILE"
    echo "Saved failed tile $TILE_ID to $FAILED_FILE"
fi


echo "Validation tile download job completed for $TILE_ID"
