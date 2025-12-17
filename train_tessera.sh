#!/bin/bash
#SBATCH --job-name=train_ssl_valid
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train_ssl_%j.out
#SBATCH --error=logs/train_ssl_%j.err


# Navigate to working directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

# Run the job
apptainer exec --nv ml_env.sif \
    python /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml/train_ssl.py \
        --index_dir /scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled \
        --data_dir /scratch/users/psinghal/time_series \
        --batch_size 2048 \
        --epochs 1

