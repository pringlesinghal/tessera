#!/bin/bash

# Multi-GPU Training Test Script
# Tests incremental functionality before full deployment

set -e  # Exit on any error

echo "=== Multi-GPU Training Test Suite ==="
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [[ ! -f "tessera_ml/train_multi_gpu_v2.py" ]]; then
    echo "Error: Must run from tessera/ directory"
    exit 1
fi

# Phase 1: Test minimal distributed communication
echo ""
echo "=== Phase 1: Testing Minimal Distributed Communication ==="
echo "Command: torchrun --nproc_per_node=2 tessera_ml/test_dist_minimal.py"

if torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 tessera_ml/test_dist_minimal.py; then
    echo "✅ Phase 1 PASSED: Basic distributed communication works"
else
    echo "❌ Phase 1 FAILED: Basic distributed communication failed"
    echo "Check NCCL configuration and GPU availability"
    exit 1
fi

# Phase 2: Test distributed initialization in our training script
echo ""
echo "=== Phase 2: Testing Distributed Initialization ==="
echo "Command: torchrun --nproc_per_node=2 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py --test_only"

if torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py --test_only; then
    echo "✅ Phase 2 PASSED: Training script distributed initialization works"
else
    echo "❌ Phase 2 FAILED: Training script distributed initialization failed"
    exit 1
fi

# Phase 3: Test basic DDP training (few batches)
echo ""
echo "=== Phase 3: Testing Basic DDP Training ==="
echo "Command: torchrun --nproc_per_node=2 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py"

if torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29502 tessera_ml/train_multi_gpu_v2.py --config tessera_ml/test_config.py; then
    echo "✅ Phase 3 PASSED: Basic DDP training works"
    echo ""
    echo "🎉 ALL TESTS PASSED! Ready to proceed with FSDP implementation"
else
    echo "❌ Phase 3 FAILED: Basic DDP training failed"
    echo "Debug the model/dataset integration"
    exit 1
fi

echo ""
echo "=== Test Summary ==="
echo "✅ Distributed communication: WORKING"
echo "✅ Training script initialization: WORKING" 
echo "✅ DDP training with TreeDataset: WORKING"
echo ""
echo "Next steps:"
echo "1. If running in container, test with: apptainer exec --nv ml_env.sif bash tessera_ml/test_multi_gpu.sh"
echo "2. Add FSDP wrapping incrementally"
echo "3. Scale up batch size and model parameters"