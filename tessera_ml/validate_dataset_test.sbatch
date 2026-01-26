#!/bin/bash
#SBATCH --job-name=validate_test
#SBATCH --output=logs/validate_test_%j.log
#SBATCH --error=logs/validate_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=serc

# Quick validation test on small subset of data
# Tests the validation approach before running full job

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 4G"
echo "Time limit: 30 minutes"
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

# Check paths exist
echo "Checking paths..."
echo "Index directory exists: $(test -d /scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled && echo 'YES' || echo 'NO')"
echo "Data directory exists: $(test -d $SCRATCH/time_series && echo 'YES' || echo 'NO')"
echo "Container exists: $(test -f /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif && echo 'YES' || echo 'NO')"
echo ""

# List sample data structure
echo "Sample data structure:"
echo "Years available: $(ls $SCRATCH/time_series/ | grep '^[0-9]' | tr '\n' ' ')"
echo "Sample parquet files: $(ls /scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled/part.*.parquet | head -3 | xargs -n1 basename | tr '\n' ' ')"
echo ""

# Run the test validation
echo "Starting validation test..."
echo "Testing 50 tiles across all years (2016-2024)"
echo "Expected runtime: 5-10 minutes (testing all 9 years)"
echo ""

time apptainer exec \
    --bind /scratch/groups/dlobell/psinghal/sentineldownloader/tessera:/scratch/groups/dlobell/psinghal/sentineldownloader/tessera \
    --bind /scratch/users/psinghal:/scratch/users/psinghal \
    --bind /scratch:/scratch \
    /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif \
    python3 validate_dataset_test.py

echo ""
echo "Validation test completed"
echo "Check the output above to verify:"
echo "1. Script can read parquet files"
echo "2. Script can access timeseries data"
echo "3. Tile validation logic works correctly"
echo "4. Sample filtering produces expected results"
echo ""
echo "If test is successful, run full validation with:"
echo "sbatch validate_dataset.sbatch"