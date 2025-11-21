#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_single_mask.sh /path/to/tile_100km_geo.tif

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 TILE_GTIFF_PATH [EXPORT_BUCKET] [EXPORT_PREFIX]"
    exit 1
fi

TILE_PATH="$1"
EXPORT_BUCKET="${2:-sidd_rajasthan}"
EXPORT_PREFIX="${3:-psinghal/farmtree_tiles/test}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/scratch/groups/dlobell/psinghal/sentineldownloader"

apptainer run "${PROJECT_ROOT}/dependencies/apptainer/tesseraenvx.sif" \
    python "${SCRIPT_DIR}/get_masks.py" \
        --tile-paths "${TILE_PATH}" \
        --export-bucket "${EXPORT_BUCKET}" \
        --export-prefix "${EXPORT_PREFIX}" \
        --task-prefix farmtree_test \
        --cloud-optimized \
        --status-path "${SCRIPT_DIR}/mask_status.jsonl"

