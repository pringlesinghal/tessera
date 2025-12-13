#!/usr/bin/env bash
#SBATCH --job-name=test_nccl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=2
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --output=logs/test_nccl_%j.log
#SBATCH --error=logs/test_nccl_%j.err

set -euo pipefail

# Configuration
NUM_GPUS=2
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"

# Change to the tessera_ml directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

echo "========================================"
echo "NCCL Test - Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $WORK_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set NCCL environment variables for Sherlock
echo "Setting NCCL environment variables..."
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # Disable InfiniBand for single-node
export NCCL_P2P_LEVEL=NVL         # Use NVLink if available, otherwise PCIe
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_NET="Socket"
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4

# Print diagnostic information
echo ""
echo "========================================"
echo "System Diagnostics"
echo "========================================"

echo "Available GPUs:"
nvidia-smi --list-gpus

echo ""
echo "GPU Topology:"
nvidia-smi topo -m || echo "GPU topology not available"

echo ""
echo "Network Interfaces:"
ip addr show | grep -E '^[0-9]+:|inet ' || ifconfig -a | grep -E 'flags|inet '

echo ""
echo "NCCL Environment Variables:"
env | grep NCCL | sort

echo ""
echo "========================================"
echo "Running NCCL Test..."
echo "========================================"

# Set PYTHONPATH
PARENT_DIR="$(dirname "$WORK_DIR")"
export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Set bind paths
BIND_PATHS="$WORK_DIR:$WORK_DIR"
BIND_PATHS="$BIND_PATHS,$PARENT_DIR:$PARENT_DIR"
BIND_PATHS="$BIND_PATHS,/scratch:/scratch"

# Run NCCL test
apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        "$WORK_DIR/test_nccl.py"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ NCCL test PASSED!"
    echo "You can now run: sbatch train_2gpu.sbatch"
else
    echo "❌ NCCL test FAILED with exit code: $EXIT_CODE"
    echo "Check the logs above for errors."
    echo "See NCCL_TROUBLESHOOTING.md for solutions."
fi
echo "========================================"

exit $EXIT_CODE

