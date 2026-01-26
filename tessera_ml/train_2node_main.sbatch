#!/usr/bin/env bash
#SBATCH --job-name=train_batch1024_2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH --partition=serc
#SBATCH --gpus-per-node=8
#SBATCH -C GPU_SKU:A100_SXM4,GPU_MEM:80GB
#SBATCH --output=logs/train_batch1024_2node_%j.log
#SBATCH --error=logs/train_batch1024_2node_%j.err

set -euo pipefail

CONFIG_FILE_ARG="${1:-configs/ssl_config_test_batch2_modified.py}"
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml
WORK_DIR="$(pwd)"

if [[ "$CONFIG_FILE_ARG" = /* ]]; then
    CONFIG_FILE="$CONFIG_FILE_ARG"
else
    CONFIG_FILE="$WORK_DIR/$CONFIG_FILE_ARG"
fi

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Node rank: $SLURM_NODEID"
echo "Config: $CONFIG_FILE"

# --------------------
# NCCL + Torch config
# --------------------
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_FAMILY=AF_INET

# Prefer InfiniBand, but don't over-constrain
export NCCL_SOCKET_IFNAME=ib0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# --------------------
# Rendezvous setup
# --------------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
NODE_RANK=$SLURM_NODEID

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$NODE_RANK"

# --------------------
# Bind paths
# --------------------
PARENT_DIR="$(dirname "$WORK_DIR")"
BIND_PATHS="$WORK_DIR:$WORK_DIR,$PARENT_DIR:$PARENT_DIR,/scratch:/scratch"

export PYTHONPATH="$PARENT_DIR:$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --------------------
# Launch (CRITICAL: srun)
# --------------------
srun apptainer exec --nv \
    --bind "$BIND_PATHS" \
    --pwd "$WORK_DIR" \
    "$CONTAINER_IMAGE" \
    torchrun \
        --nnodes=2 \
        --nproc_per_node=8 \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$WORK_DIR/train_ssl_multi_gpu.py" \
        --config "$CONFIG_FILE"
