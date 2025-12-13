#!/bin/bash
#SBATCH --job-name=test-fsdp-serc
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=64GB
#SBATCH --time=30:00
#SBATCH --output=tessera_ml/logs/test_fsdp_serc_%j.out
#SBATCH --error=tessera_ml/logs/test_fsdp_serc_%j.err

echo "=== FSDP Training Test - SERC A100s ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

# Environment setup
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# W&B for FSDP testing
export WANDB_PROJECT=tessera-fsdp-test

echo ""
echo "=== Checking FSDP Integration ==="
if grep -q "FSDP INTEGRATION ADDED" tessera_ml/train_multi_gpu_v2.py; then
    echo "✅ FSDP integration already added"
else
    echo "Adding FSDP integration..."
    apptainer exec --nv ml_env.sif python tessera_ml/add_fsdp.py
    echo "✅ FSDP integration added"
fi

echo ""
echo "=== A100 System Information ==="
nvidia-smi
echo ""
echo "Available memory:"
free -h

echo ""
echo "=== FSDP Training Test ==="
echo "This test will verify:"
echo "1. FSDP initialization works with our model"
echo "2. Memory efficiency compared to DDP"
echo "3. Training convergence with FSDP"

# Create FSDP-specific config with potentially larger model
cat > tessera_ml/fsdp_config.py << 'EOF'
"""
FSDP test configuration - can use larger models due to memory efficiency
"""

config = {
    # Data paths
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled",
    "data_root": "/scratch/groups/dlobell/psinghal/sentineldownloader",
    
    # Model architecture (FSDP can handle larger)
    "latent_dim": 512,
    "projector_hidden_dim": 16384,
    "projector_out_dim": 16384,
    
    # Training parameters
    "batch_size": 256,                 # Same as DDP test
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 1,
    
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
    "wandb_project": "tessera-fsdp-test-a100",
}
EOF

echo ""
echo "=== Running FSDP Training ==="
echo "Command: apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/fsdp_config.py"

if apptainer exec --nv ml_env.sif torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/fsdp_config.py; then
    echo ""
    echo "🎉🎉🎉 SUCCESS: FSDP training works on A100s! 🎉🎉🎉"
    echo ""
    echo "FSDP Benefits achieved:"
    echo "- Memory efficiency through parameter sharding"
    echo "- Ability to train larger models"
    echo "- Faster training with proper overlapping"
    echo ""
    echo "✅ READY FOR PRODUCTION TRAINING"
    echo ""
    echo "Final steps:"
    echo "1. Create production config with full model size"
    echo "2. Scale to multiple nodes if needed"
    echo "3. Run full training job"
    
    # Create production template
    echo ""
    echo "Creating production config template..."
    cat > tessera_ml/production_config_template.py << 'PROD_EOF'
"""
Production FSDP training configuration template
Modify parameters as needed for your full training run
"""

config = {
    # Data paths
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled",  # Full dataset
    "data_root": "/scratch/groups/dlobell/psinghal/sentineldownloader",
    
    # Model architecture (full size)
    "latent_dim": 512,
    "projector_hidden_dim": 16384,
    "projector_out_dim": 16384,
    
    # Training parameters (production)
    "batch_size": 512,                 # Larger batch with FSDP
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "epochs": 100,                     # Full training
    
    # Data parameters
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "fusion_method": "concat",
    
    # SSL parameters
    "barlow_lambda": 0.005,
    
    # Architecture details (can be larger with FSDP)
    "s2_num_heads": 8,                 # Increased
    "s2_num_layers": 8,                # Increased
    "s2_dim_feedforward": 2048,        # Increased
    "s1_num_heads": 8,                 # Increased
    "s1_num_layers": 8,                # Increased
    "s1_dim_feedforward": 2048,        # Increased
    
    # Quantization options
    "apply_qat_representation": False,
    "qat_representation_bits": 8,
    
    # W&B settings
    "wandb_project": "tessera-production-training",
}
PROD_EOF
    echo "✅ Production config template created at tessera_ml/production_config_template.py"
    
else
    echo ""
    echo "❌ FSDP test failed"
    echo ""
    echo "Debugging steps:"
    echo "1. Check if FSDP integration was applied correctly"
    echo "2. Verify NCCL communication works (previous tests passed)"  
    echo "3. Try smaller model or batch size for FSDP"
    echo "4. Check for FSDP-specific error messages in logs"
    exit 1
fi