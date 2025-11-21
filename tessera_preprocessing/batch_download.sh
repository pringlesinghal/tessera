#!/usr/bin/env bash
#SBATCH --job-name=tile_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=serc
#SBATCH --output=logs/tile_dl_%A_%a_%j.log
#SBATCH --error=logs/tile_dl_%A_%a_%j.err
#SBATCH --array=0-999

set -euo pipefail
# submitted 0, 1000, 2000, 3000, 4000 (due)

# File containing the tile IDs
TILE_LIST="${1:-/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles/subtile_tile_list_shuffled.txt}"
STARTED_FILE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles/started_tiles.txt"
COMPLETED_FILE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles/completed_tiles.txt"
COMPLETED_UPLOAD_FILE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles/completed_upload_tiles.txt"

# Change to preprocessing directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_preprocessing

# Get the tile ID for this array task (line number based on array index)
TILE_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TILE_LIST")
echo "Processing tile: $TILE_ID (array task $SLURM_ARRAY_TASK_ID)"

echo "$TILE_ID" >> "$STARTED_FILE"
echo "Saved started tile $TILE_ID to $STARTED_FILE"

# Loop sequentially over years 2024 → 2016
for YEAR in {2024..2016..-1}; do
    echo "Processing year: $YEAR for tile $TILE_ID"
    bash s1_s2_downloader.sh "$TILE_ID" "$YEAR"
    bash s1_s2_stacker.sh "$TILE_ID" "$YEAR"
RAW_DIR="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/time_series_sparse/${YEAR}/${TILE_ID}/data_raw"
RAW_DIR_SAR="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/time_series_sparse/${YEAR}/${TILE_ID}/data_sar_raw"
LOG_DIR="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/time_series_sparse/${YEAR}/${TILE_ID}/logs"
    if [ -d "$RAW_DIR" ]; then
        rm -rf "$RAW_DIR"
        echo "Deleted raw directory $RAW_DIR"
    fi
    if [ -d "$RAW_DIR_SAR" ]; then
        rm -rf "$RAW_DIR_SAR"
        echo "Deleted raw SAR directory $RAW_DIR_SAR"
    fi
    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
        echo "Deleted log directory $LOG_DIR"
    fi
done

echo "Finished tile: $TILE_ID"

# Record successful completion
echo "$TILE_ID" >> "$COMPLETED_FILE"
echo "Saved completed tile $TILE_ID to $COMPLETED_FILE"


# export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
# GCS_BUCKET="gs://sidd_rajasthan/psinghal/time_series_sparse"
# BASE_DIR="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera"
# TIMESERIES_DIR="${BASE_DIR}/time_series_sparse"

# echo "Uploading data for $TILE_ID to $GCS_BUCKET"
# UPLOAD_SUCCESS=true

# for YEAR in {2024..2016..-1}; do
#     SRC_PATH="${TIMESERIES_DIR}/${YEAR}/${TILE_ID}"
#     DEST_PATH="${GCS_BUCKET}/${YEAR}/${TILE_ID}/"

#     if [ -d "$SRC_PATH" ]; then
#         echo "Uploading $SRC_PATH → $DEST_PATH"
#         if gsutil cp -r "$SRC_PATH" "$DEST_PATH"; then
#             echo "Upload successful: $SRC_PATH"
#             rm -rf "$SRC_PATH"
#             echo "Deleted local folder: $SRC_PATH"
#         else
#             echo "❌ Upload failed for $SRC_PATH"
#             UPLOAD_SUCCESS=false
#         fi
#     else
#         echo "Warning: $SRC_PATH not found, skipping."
#     fi
# done

# if [ "$UPLOAD_SUCCESS" = true ]; then
#     echo "$TILE_ID" >> "$COMPLETED_UPLOAD_FILE"
#     echo "✅ Finished and recorded tile: $TILE_ID"
# else
#     echo "⚠️ Upload failed for one or more years of $TILE_ID — keeping local data."
# fi
