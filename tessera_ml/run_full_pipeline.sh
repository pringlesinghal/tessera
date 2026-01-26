#!/usr/bin/env bash
#
# Run the full SSL data curation pipeline
#
# This script submits all three jobs in sequence with proper dependencies:
#   1. extract_embeddings - Extract 128-dim embeddings from 50M samples
#   2. run_clustering - Hierarchical k-means clustering (4 levels)
#   3. sample_curated_index - Sample from clusters to create curated dataset
#
# Usage:
#   ./run_full_pipeline.sh [target_size]
#
# Example:
#   ./run_full_pipeline.sh 1000000    # Create 1M sample curated dataset
#   ./run_full_pipeline.sh 5000000    # Create 5M sample curated dataset
#

set -euo pipefail

# Configuration
TARGET_SIZE="${1:-1000000}"  # Default: 1M samples
OUTPUT_DIR="curation_outputs"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml

echo "========================================"
echo "SSL Data Curation Pipeline"
echo "========================================"
echo "Target curated dataset size: ${TARGET_SIZE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================"
echo ""

# Step 1: Submit embedding extraction
echo "Submitting Step 1: Embedding extraction (50M samples)..."
JOB1_OUTPUT=$(sbatch extract_embeddings.sbatch)
JOB1_ID=$(echo "$JOB1_OUTPUT" | awk '{print $4}')
echo "  Job ID: $JOB1_ID"
echo "  Monitor: squeue -j $JOB1_ID"
echo "  Logs: logs/extract_embeddings_${JOB1_ID}.log"
echo ""

# Step 2: Submit clustering (depends on Step 1)
echo "Submitting Step 2: Hierarchical clustering (depends on Job $JOB1_ID)..."
JOB2_OUTPUT=$(sbatch --dependency=afterok:${JOB1_ID} run_clustering.sbatch)
JOB2_ID=$(echo "$JOB2_OUTPUT" | awk '{print $4}')
echo "  Job ID: $JOB2_ID"
echo "  Monitor: squeue -j $JOB2_ID"
echo "  Logs: logs/run_clustering_${JOB2_ID}.log"
echo ""

# Step 3: Submit sampling (depends on Step 2)
echo "Submitting Step 3: Curated index sampling (depends on Job $JOB2_ID)..."
JOB3_OUTPUT=$(sbatch --dependency=afterok:${JOB2_ID} sample_curated_index.sbatch ${TARGET_SIZE})
JOB3_ID=$(echo "$JOB3_OUTPUT" | awk '{print $4}')
echo "  Job ID: $JOB3_ID"
echo "  Monitor: squeue -j $JOB3_ID"
echo "  Logs: logs/sample_curated_${JOB3_ID}.log"
echo ""

echo "========================================"
echo "Pipeline submitted successfully!"
echo "========================================"
echo ""
echo "Job chain: $JOB1_ID -> $JOB2_ID -> $JOB3_ID"
echo ""
echo "Monitor all jobs:"
echo "  squeue -j ${JOB1_ID},${JOB2_ID},${JOB3_ID}"
echo ""
echo "Or watch progress:"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "Expected timeline:"
echo "  Step 1 (extraction): ~12-24 hours for 50M samples"
echo "  Step 2 (clustering): ~6-12 hours"
echo "  Step 3 (sampling):   ~1-2 hours"
echo ""
echo "Final output will be in:"
echo "  ${OUTPUT_DIR}/curated_index_${TARGET_SIZE}/"
echo ""
echo "To use the curated dataset for training:"
echo "  Update config: index_dir = '${OUTPUT_DIR}/curated_index_${TARGET_SIZE}'"
echo "========================================"
