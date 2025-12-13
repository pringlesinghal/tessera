#!/bin/bash
#SBATCH --job-name=test-ddp-dev
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=32GB
#SBATCH --time=15:00
#SBATCH --output=tessera_ml/logs/test_ddp_dev_%j.out
#SBATCH --error=tessera_ml/logs/test_ddp_dev_%j.err

echo "=== DDP Training Test - Dev Partition ==="
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

# Disable W&B for testing
export WANDB_MODE=disabled

echo ""
echo "=== System Information ==="
nvidia-smi
echo "Available memory:"
free -h

echo ""
echo "=== DDP Training Test (Small Scale) ==="
echo "Command: apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py"

# Run the actual training test
if apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py; then
    echo ""
    echo "✅ DDP Training Test PASSED on dev partition!"
    echo ""
    echo "🎉 SUCCESS: Ready to scale up to A100s on serc partition"
    echo ""
    echo "Next steps:"
    echo "1. Submit test_ddp_serc.sbatch for A100 testing"
    echo "2. If A100 DDP works, add FSDP with python tessera_ml/add_fsdp.py"
    echo "3. Scale up batch size and model complexity"
    
else
    echo ""
    echo "❌ DDP Training Test FAILED"
    echo ""
    echo "Debugging steps:"
    echo "1. Check GPU memory usage during training"
    echo "2. Verify dataset loading works"
    echo "3. Test with even smaller batch size"
    
    # Quick debugging info
    echo ""
    echo "=== Debug Information ==="
    apptainer exec --nv ml_env.sif python -c "
import torch
print(f'GPU memory before any allocation:')
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    print(f'  GPU {i}: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, {torch.cuda.memory_reserved()/1024**2:.1f} MB reserved')
"
    exit 1
fi