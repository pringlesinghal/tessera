#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_masks.sh [EXPORT_BUCKET] [EXPORT_PREFIX]

EXPORT_BUCKET="${1:-sidd_rajasthan}"
EXPORT_PREFIX="${2:-psinghal/farmtree_tiles/full}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/scratch/groups/dlobell/psinghal/sentineldownloader"
TILE_LIST="${PROJECT_ROOT}/tessera/shapefiles/india_tiles/mgrs_tiles/parent_tile_list.txt"
SHUFFLED_LIST="$(mktemp)"

if [[ ! -f "${TILE_LIST}" ]]; then
    echo "Parent tile list not found: ${TILE_LIST}"
    exit 1
fi

shuf "${TILE_LIST}" > "${SHUFFLED_LIST}"
echo "Shuffled tile list written to ${SHUFFLED_LIST}"

apptainer run "${PROJECT_ROOT}/dependencies/apptainer/tesseraenvx.sif" \
    python "${SCRIPT_DIR}/get_masks.py" \
        --tile-list-file "${SHUFFLED_LIST}" \
        --export-bucket "${EXPORT_BUCKET}" \
        --export-prefix "${EXPORT_PREFIX}" \
        --task-prefix farmtree_batch \
        --cloud-optimized \
        --status-path "${SCRIPT_DIR}/mask_status.jsonl"

rm -f "${SHUFFLED_LIST}"

