#!/bin/bash
#SBATCH --job-name=validate_dataset
#SBATCH --output=logs/validate_dataset_%j.log
#SBATCH --error=logs/validate_dataset_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=serc

# Dataset validation for 200M+ tree pixels across ~80TB
# Efficiently identifies valid tiles across all years 2016-2024

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 64GB"
echo "Container: ml_env.sif"
echo "========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export SCRATCH=/scratch/users/psinghal
export PYTHONPATH=/scratch/groups/dlobell/psinghal/sentineldownloader/tessera:/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml

# Show system info
echo "Available memory:"
free -h
echo ""
echo "CPU info:"
lscpu | grep "CPU(s):"
echo ""

# Run the validation script with container
echo "Starting dataset validation..."
echo "Expected to process ~200M samples across 178 parquet files"
echo "Validating tiles across years 2016-2024"
echo ""

time apptainer exec \
    --bind /scratch/groups/dlobell/psinghal/sentineldownloader/tessera:/scratch/groups/dlobell/psinghal/sentineldownloader/tessera \
    --bind /scratch/users/psinghal:/scratch/users/psinghal \
    --bind /scratch:/scratch \
    /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif \
    python3 validate_dataset_optimized.py

echo "Dataset validation completed"
echo "Check output directory: /scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid"