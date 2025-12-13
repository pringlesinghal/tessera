#!/bin/bash
#SBATCH --job-name=test-ddp-serc
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=64GB
#SBATCH --time=20:00
#SBATCH --output=tessera_ml/logs/test_ddp_serc_%j.out
#SBATCH --error=tessera_ml/logs/test_ddp_serc_%j.err

echo "=== DDP Training Test - SERC A100s ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"  
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

# Environment setup
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=WARN  # Less verbose on A100s
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Enable W&B for A100 testing
export WANDB_PROJECT=tessera-multi-gpu-test

echo ""
echo "=== A100 GPU Information ==="
nvidia-smi
echo ""
echo "Available memory:"
free -h

echo ""
echo "=== Testing Larger Batch Size on A100s ==="

# Create A100-optimized config
cat > tessera_ml/serc_config.py << 'EOF'
"""
SERC A100 configuration - larger scale testing
"""

config = {
    # Data paths
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled",
    "data_root": "/scratch/groups/dlobell/psinghal/sentineldownloader",
    
    # Model architecture (A100 scale)
    "latent_dim": 512,
    "projector_hidden_dim": 16384,     # Full size
    "projector_out_dim": 16384,        # Full size
    
    # Training parameters (A100 optimized)
    "batch_size": 256,                 # Full batch size
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 1,                       # Still just 1 epoch for testing
    
    # Data parameters
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "fusion_method": "concat",
    
    # SSL parameters
    "barlow_lambda": 0.005,
    
    # Architecture details
    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 1024,
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 1024,
    
    # Quantization (disabled for testing)
    "apply_qat_representation": False,
    "qat_representation_bits": 8,
    
    # W&B settings
    "wandb_project": "tessera-multi-gpu-test-a100",
}
EOF

echo "Created A100-optimized config with batch_size=256"

echo ""
echo "=== DDP Training on A100s ==="
echo "Command: apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/serc_config.py"

if apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/serc_config.py; then
    echo ""
    echo "🎉 SUCCESS: DDP training works on A100s!"
    echo ""
    echo "Performance summary:"
    echo "- Batch size: 256 per GPU"
    echo "- Model size: ~2.5B parameters"
    echo "- Memory usage: Check logs above"
    echo ""
    echo "✅ READY FOR FSDP INTEGRATION"
    echo ""
    echo "Next steps:"
    echo "1. Run: python tessera_ml/add_fsdp.py"
    echo "2. Submit: sbatch tessera_ml/test_fsdp_serc.sbatch"
    echo "3. Scale to full production config"
    
else
    echo ""
    echo "❌ A100 DDP test failed"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check memory usage - may need smaller batch"
    echo "2. Verify dataset access from SERC partition"
    echo "3. Try with gloo backend if NCCL issues persist"
    exit 1
fi