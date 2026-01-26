#!/usr/bin/env bash
#
# Find the optimal batch size for embedding extraction.
# Uses binary search to find the largest batch size that fits in GPU memory.
#
# Usage: ./scripts/find_optimal_batch_size.sh
#
# Run this from an interactive session with GPUs:
#   srun --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=512G --time=48:00:00 \
#        --partition=serc --gpus-per-node=8 -C GPU_SKU:A100_SXM4,GPU_MEM:80GB \
#        --pty bash
#   ./scripts/find_optimal_batch_size.sh

set -euo pipefail

# Configuration
WORK_DIR="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml"
CONTAINER_IMAGE="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"
CHECKPOINT="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera_best_model_fsdp_20250427_084307.pt"
TEST_OUTPUT_DIR="${WORK_DIR}/batch_size_test_outputs"

# Batch size range (per-GPU)
MIN_BATCH=256
MAX_BATCH=8192
CURRENT_BATCH=4096  # Start in the middle-high range

# Track results
declare -A RESULTS
BEST_WORKING=0
SMALLEST_FAILED=999999

# Number of batches to test (small number for quick testing)
TEST_BATCHES=5

cd "$WORK_DIR"
PARENT_DIR="$(dirname $WORK_DIR)"

# Setup environment
export PYTHONPATH="$PARENT_DIR:$WORK_DIR"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

BIND_PATHS="$WORK_DIR:$WORK_DIR,$PARENT_DIR:$PARENT_DIR,/scratch:/scratch"

# Clean up test directory
rm -rf "$TEST_OUTPUT_DIR"
mkdir -p "$TEST_OUTPUT_DIR"

echo "========================================"
echo "Batch Size Optimization Script"
echo "========================================"
echo "Container: $CONTAINER_IMAGE"
echo "Checkpoint: $CHECKPOINT"
echo "Test output: $TEST_OUTPUT_DIR"
echo "Batch size range: $MIN_BATCH - $MAX_BATCH (per-GPU)"
echo "========================================"
echo ""

# Get number of GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $N_GPUS GPUs"
echo ""

# Function to test a batch size
test_batch_size() {
    local batch_size=$1
    local effective_batch=$((batch_size * N_GPUS))
    local test_samples=$((effective_batch * TEST_BATCHES))

    echo "----------------------------------------"
    echo "Testing batch_size=$batch_size (effective=$effective_batch)"
    echo "Will process $test_samples samples in $TEST_BATCHES batches"
    echo "----------------------------------------"

    # Clean previous test output
    rm -rf "${TEST_OUTPUT_DIR}/embeddings.npy" "${TEST_OUTPUT_DIR}/extraction_checkpoint.json"

    # Create a small test script that runs for limited batches
    local test_script="${TEST_OUTPUT_DIR}/test_extraction.py"
    cat > "$test_script" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Quick batch size test - runs only a few batches."""
import sys
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TreeDataset
from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel

def infer_config_from_state_dict(state_dict):
    config = {
        "latent_dim": 128, "fusion_method": "concat",
        "sample_size_s2": 40, "sample_size_s1": 40,
    }
    s2_layers = set()
    s1_layers = set()
    for key in state_dict.keys():
        if "s2_backbone.transformer_encoder.layers." in key:
            s2_layers.add(int(key.split(".")[3]))
        if "s1_backbone.transformer_encoder.layers." in key:
            s1_layers.add(int(key.split(".")[3]))
    config["s2_num_layers"] = len(s2_layers) if s2_layers else 4
    config["s1_num_layers"] = len(s1_layers) if s1_layers else 4
    for key, value in state_dict.items():
        if "s2_backbone.transformer_encoder.layers.0.self_attn.in_proj_weight" in key:
            d_model = value.shape[1]
            config["latent_dim"] = d_model // 4
            config["s2_num_heads"] = 8 if d_model >= 512 else 4
            config["s1_num_heads"] = config["s2_num_heads"]
            break
    for key, value in state_dict.items():
        if "s2_backbone.transformer_encoder.layers.0.linear1.weight" in key:
            config["s2_dim_feedforward"] = value.shape[0]
            config["s1_dim_feedforward"] = value.shape[0]
            break
    projector_layers = sorted([
        int(key.split(".")[2]) for key in state_dict.keys()
        if key.startswith("projector.net.") and ".weight" in key and "running" not in key
    ])
    if projector_layers:
        last_layer = projector_layers[-1]
        for key, value in state_dict.items():
            if f"projector.net.{last_layer}.weight" in key:
                config["projector_out_dim"] = value.shape[0]
                break
        for key, value in state_dict.items():
            if "projector.net.0.weight" in key:
                config["projector_hidden_dim"] = value.shape[0]
                break
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda")
    print(f"Using {n_gpus} GPUs")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("_fsdp_wrapped_module."):
            new_key = key[len("_fsdp_wrapped_module."):]
        elif key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
        new_state_dict[new_key] = value

    config = infer_config_from_state_dict(new_state_dict)
    latent_dim = config.get("latent_dim", 128)

    # Create model
    s2_enc = TransformerEncoder(
        band_num=10, latent_dim=latent_dim,
        nhead=config.get("s2_num_heads", 8),
        num_encoder_layers=config.get("s2_num_layers", 8),
        dim_feedforward=config.get("s2_dim_feedforward", 2048),
        max_seq_len=config.get("sample_size_s2", 40),
    ).to(device)

    s1_enc = TransformerEncoder(
        band_num=2, latent_dim=latent_dim,
        nhead=config.get("s1_num_heads", 8),
        num_encoder_layers=config.get("s1_num_layers", 8),
        dim_feedforward=config.get("s1_dim_feedforward", 2048),
        max_seq_len=config.get("sample_size_s1", 40),
    ).to(device)

    projector = ProjectionHead(
        input_dim=latent_dim,
        output_dim=config.get("projector_out_dim", 8192),
        hidden_dim=config.get("projector_hidden_dim", 8192),
    ).to(device)

    model = MultimodalBTModel(
        s2_enc, s1_enc, projector,
        fusion_method=config.get("fusion_method", "concat"),
        return_repr=True,
        latent_dim=latent_dim,
        apply_qat_representation=False,
    ).to(device)

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    if n_gpus > 1:
        print(f"Wrapping with DataParallel")
        model = nn.DataParallel(model)

    # Create dataset
    print("Creating dataset...")
    dataset = TreeDataset(
        index_dir="/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid",
        data_dir="/scratch/users/psinghal/time_series",
        year=None,
        years=list(range(2016, 2025)),
        cache_size=50,
        sample_size_s2=40,
        sample_size_s1=40,
        normalize=True,
    )

    effective_batch = args.batch_size * n_gpus
    test_samples = effective_batch * args.num_batches

    # Use subset for testing
    indices = list(range(min(test_samples, len(dataset))))
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )

    print(f"Testing with batch_size={args.batch_size} (effective={effective_batch})")
    print(f"Running {args.num_batches} batches ({test_samples} samples)")

    # Warmup
    torch.cuda.empty_cache()

    # Run test
    times = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break

            start = time.time()

            s2_data = batch["s2_aug1"].to(device)
            s1_data = batch["s1_aug1"].to(device)

            _, embeddings = model(s2_data, s1_data)

            # Force sync
            torch.cuda.synchronize()

            elapsed = time.time() - start
            times.append(elapsed)

            samples_per_sec = effective_batch / elapsed

            # Get memory usage
            mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            print(f"  Batch {batch_idx+1}/{args.num_batches}: "
                  f"{elapsed:.2f}s, {samples_per_sec:.0f} samples/sec, "
                  f"GPU mem: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

    avg_time = sum(times) / len(times)
    avg_throughput = effective_batch / avg_time

    print(f"\n{'='*50}")
    print(f"SUCCESS: batch_size={args.batch_size} works!")
    print(f"Average: {avg_time:.2f}s/batch, {avg_throughput:.0f} samples/sec")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/(1024**3):.1f}GB")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
PYTHON_EOF

    # Run the test
    local start_time=$(date +%s)

    if apptainer exec --nv \
        --bind "$BIND_PATHS" \
        --pwd "$WORK_DIR" \
        --env PYTHONPATH="$PYTHONPATH" \
        --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
        --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
        --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
        "$CONTAINER_IMAGE" \
        python "$test_script" \
            --checkpoint "$CHECKPOINT" \
            --batch_size "$batch_size" \
            --num_batches "$TEST_BATCHES" \
            --num_workers 8 \
        2>&1 | tee "${TEST_OUTPUT_DIR}/test_batch_${batch_size}.log"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        echo ""
        echo "SUCCESS: batch_size=$batch_size completed in ${duration}s"
        RESULTS[$batch_size]="SUCCESS"
        return 0
    else
        echo ""
        echo "FAILED: batch_size=$batch_size (likely OOM)"
        RESULTS[$batch_size]="FAILED"
        return 1
    fi
}

# Binary search for optimal batch size
echo "Starting binary search for optimal batch size..."
echo ""

LOW=$MIN_BATCH
HIGH=$MAX_BATCH

while [ $LOW -le $HIGH ]; do
    MID=$(( (LOW + HIGH) / 2 ))
    # Round to nearest 128 for cleaner numbers
    MID=$(( (MID / 128) * 128 ))

    if [ $MID -lt $MIN_BATCH ]; then
        MID=$MIN_BATCH
    fi

    echo ""
    echo "========================================"
    echo "Binary search: LOW=$LOW, HIGH=$HIGH, testing MID=$MID"
    echo "========================================"

    if test_batch_size $MID; then
        # Success - try larger
        if [ $MID -gt $BEST_WORKING ]; then
            BEST_WORKING=$MID
        fi
        LOW=$((MID + 128))
    else
        # Failed - try smaller
        if [ $MID -lt $SMALLEST_FAILED ]; then
            SMALLEST_FAILED=$MID
        fi
        HIGH=$((MID - 128))
    fi

    # Early termination if we've narrowed it down
    if [ $((HIGH - LOW)) -lt 128 ]; then
        break
    fi
done

# Final verification of best working batch size
echo ""
echo "========================================"
echo "Final verification of best batch size: $BEST_WORKING"
echo "========================================"

if [ $BEST_WORKING -gt 0 ]; then
    test_batch_size $BEST_WORKING
fi

# Print summary
echo ""
echo "========================================"
echo "OPTIMIZATION COMPLETE"
echo "========================================"
echo ""
echo "Results summary:"
for batch_size in $(echo "${!RESULTS[@]}" | tr ' ' '\n' | sort -n); do
    echo "  batch_size=$batch_size: ${RESULTS[$batch_size]}"
done
echo ""
echo "========================================"
echo "RECOMMENDED SETTINGS:"
echo "========================================"
echo "  --batch_size $BEST_WORKING  (per-GPU)"
echo "  Effective batch size: $((BEST_WORKING * N_GPUS))"
echo ""
echo "Update extract_embeddings.sbatch with:"
echo "  --batch_size $BEST_WORKING"
echo "========================================"

# Save recommendation to file
cat > "${TEST_OUTPUT_DIR}/recommended_batch_size.txt" << EOF
# Batch size optimization results
# Generated: $(date)
# GPUs: $N_GPUS

RECOMMENDED_BATCH_SIZE=$BEST_WORKING
EFFECTIVE_BATCH_SIZE=$((BEST_WORKING * N_GPUS))

# Test results:
$(for batch_size in $(echo "${!RESULTS[@]}" | tr ' ' '\n' | sort -n); do
    echo "BATCH_${batch_size}=${RESULTS[$batch_size]}"
done)
EOF

echo ""
echo "Results saved to: ${TEST_OUTPUT_DIR}/recommended_batch_size.txt"
