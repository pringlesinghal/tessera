#!/bin/bash
#SBATCH --job-name=test-single-dev
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=8GB
#SBATCH --time=10:00
#SBATCH --output=tessera_ml/logs/test_single_dev_%j.out
#SBATCH --error=tessera_ml/logs/test_single_dev_%j.err

echo "=== Single GPU Test - Dev Partition ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

echo ""
echo "=== GPU Information ==="
nvidia-smi

echo ""
echo "=== Test: Single GPU Functionality ==="

# Test basic PyTorch GPU functionality
apptainer exec --nv ml_env.sif python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    
    # Test basic operations
    device = torch.device('cuda:0')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    print(f'✅ Basic GPU operations work')
    print(f'GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB')
    
    # Test our model imports work on GPU
    import sys
    sys.path.insert(0, '/scratch/groups/dlobell/psinghal/sentineldownloader')
    from tessera_ml.models.modules import TransformerEncoder
    
    # Create small test model
    model = TransformerEncoder(
        band_num=10, 
        latent_dim=64,  # Small for testing
        nhead=2, 
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_seq_len=10
    ).to(device)
    
    print(f'✅ Model creation on GPU successful')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    test_input = torch.randn(4, 10, 10).to(device)  # batch=4, seq=10, features=10
    with torch.no_grad():
        output = model(test_input)
    print(f'✅ Forward pass successful')
    print(f'Output shape: {output.shape}')
    print(f'GPU memory after model: {torch.cuda.memory_allocated()/1024**2:.1f} MB')
    
    print('🎉 Single GPU test PASSED!')
else:
    print('❌ CUDA not available')
    exit(1)
"

echo ""
echo "=== Single GPU Test Complete ==="