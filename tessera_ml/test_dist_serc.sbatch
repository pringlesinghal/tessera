#!/bin/bash
#SBATCH --job-name=test-dist-serc
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=32GB
#SBATCH --time=15:00
#SBATCH --output=tessera_ml/logs/test_dist_serc_%j.out
#SBATCH --error=tessera_ml/logs/test_dist_serc_%j.err

echo "=== Distributed Communication Test - SERC A100s ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

# Environment setup
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo ""
echo "=== A100 GPU Information ==="
nvidia-smi

echo ""
echo "=== Test 1: Minimal Distributed Communication ==="
echo "Command: apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 tessera_ml/test_dist_minimal.py"

if apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 tessera_ml/test_dist_minimal.py; then
    echo "✅ PHASE 1 PASSED: Basic distributed communication works on A100s"
    
    echo ""
    echo "=== Test 2: Training Script Distributed Init ==="
    echo "Command: apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py --test_only"
    
    if apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py --test_only; then
        echo "✅ PHASE 2 PASSED: Training script distributed init works on A100s"
        echo ""
        echo "🎉 A100 distributed communication tests PASSED!"
        echo "Ready for DDP training test on A100s"
    else
        echo "❌ PHASE 2 FAILED: Training script distributed init failed"
        exit 1
    fi
else
    echo "❌ PHASE 1 FAILED: Basic distributed communication failed"
    echo ""
    echo "Debugging info:"
    apptainer exec --nv ml_env.sif python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
    exit 1
fi