#!/bin/bash
#SBATCH --job-name=verify-container
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=8GB
#SBATCH --time=5:00
#SBATCH --output=tessera_ml/logs/verify_container_%j.out
#SBATCH --error=tessera_ml/logs/verify_container_%j.err

echo "=== Container Verification ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera

# Create logs directory
mkdir -p tessera_ml/logs

echo ""
echo "=== Checking Container ==="
if [[ -f "ml_env.sif" ]]; then
    echo "✅ Container found: ml_env.sif"
    ls -lh ml_env.sif
else
    echo "❌ Container not found!"
    exit 1
fi

echo ""
echo "=== Testing Container Basic Functionality ==="
apptainer exec --nv ml_env.sif python -c "
import sys
print(f'Python version: {sys.version}')

# Test PyTorch
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else \"None\"}')
    print('✅ PyTorch OK')
except ImportError as e:
    print(f'❌ PyTorch failed: {e}')
    exit(1)

# Test torchrun
try:
    import torch.distributed
    print('✅ torch.distributed OK')
except ImportError as e:
    print(f'❌ torch.distributed failed: {e}')
    exit(1)

# Test our imports
try:
    import pandas, numpy, wandb
    print('✅ Basic packages OK')
except ImportError as e:
    print(f'❌ Basic packages failed: {e}')
    exit(1)
"

echo ""
echo "=== Testing torchrun Command ==="
if apptainer exec --nv ml_env.sif which torchrun > /dev/null 2>&1; then
    echo "✅ torchrun found"
    apptainer exec --nv ml_env.sif torchrun --help | head -5
else
    echo "❌ torchrun not found"
    # Try alternative locations
    apptainer exec --nv ml_env.sif find /opt/conda -name "torchrun" 2>/dev/null || echo "torchrun not in /opt/conda"
fi

echo ""
echo "=== Testing Our Module Imports ==="
apptainer exec --nv ml_env.sif python -c "
import sys
sys.path.insert(0, '/scratch/groups/dlobell/psinghal/sentineldownloader')
try:
    from tessera_ml.dataset import TreeDataset
    print('✅ TreeDataset import OK')
except ImportError as e:
    print(f'❌ TreeDataset import failed: {e}')

try:
    from tessera_ml.models.modules import TransformerEncoder
    print('✅ TransformerEncoder import OK')  
except ImportError as e:
    print(f'❌ TransformerEncoder import failed: {e}')

try:
    from tessera_ml.models.ssl_model import MultimodalBTModel
    print('✅ MultimodalBTModel import OK')
except ImportError as e:
    print(f'❌ MultimodalBTModel import failed: {e}')
"

echo ""
echo "=== Container Verification Complete ==="