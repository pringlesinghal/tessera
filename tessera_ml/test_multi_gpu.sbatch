#!/bin/bash
#SBATCH --job-name=test-multi-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=32GB
#SBATCH --time=30:00
#SBATCH --output=tessera_ml/logs/test_multi_gpu_%j.out
#SBATCH --error=tessera_ml/logs/test_multi_gpu_%j.err

# Test Multi-GPU Training Setup
echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Date: $(date)"

# Environment setup
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NCCL configuration for Sherlock
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Check GPU availability
echo ""
echo "=== GPU Information ==="
nvidia-smi

# Navigate to project directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

# Create logs directory if it doesn't exist
mkdir -p tessera_ml/logs

# Option 1: Test without container (if possible)
echo ""
echo "=== Testing Without Container ==="
if command -v python &> /dev/null; then
    echo "Python available, testing directly..."
    bash tessera_ml/test_multi_gpu.sh
else
    echo "Python not available, will use container..."
fi

# Option 2: Test with container
echo ""
echo "=== Testing With Container ==="
if [[ -f "ml_env.sif" ]]; then
    echo "Container found, running tests..."
    apptainer exec --nv ml_env.sif bash tessera_ml/test_multi_gpu.sh
else
    echo "Container not found at ml_env.sif"
    ls -la *.sif
fi

echo ""
echo "=== Job Completed ==="
echo "Check output logs in tessera_ml/logs/"