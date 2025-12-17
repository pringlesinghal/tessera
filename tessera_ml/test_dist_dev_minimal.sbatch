#!/bin/bash
#SBATCH --job-name=test-dist-minimal
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=8GB
#SBATCH --time=5:00
#SBATCH --output=tessera_ml/logs/test_dist_minimal_%j.out
#SBATCH --error=tessera_ml/logs/test_dist_minimal_%j.err

echo "=== Minimal 2-GPU Test - Dev Partition ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

# Environment setup
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo ""
echo "=== GPU Information ==="
nvidia-smi

echo ""
echo "=== Minimal Distributed Communication Test ==="
apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 tessera_ml/test_dist_minimal.py