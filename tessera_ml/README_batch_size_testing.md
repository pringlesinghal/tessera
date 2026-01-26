# Batch Size Testing Results

## Summary

Successfully tested training with different batch sizes on 8x A100 80GB GPUs. Found that **batch size 512 per GPU** works without OOM errors.

## Environment

- **Hardware:** 8x NVIDIA A100-SXM4-80GB GPUs
- **Node:** sh03-17n03 (serc partition)
- **Container:** `../ml_env.sif` (Apptainer/Singularity)
- **Training Script:** `train_ssl_multi_gpu.py`

## Test Results

| Batch Size per GPU | Global Batch Size | Status | Notes |
|---------------------|------------------|---------|--------|
| 1024 | 8192 | ❌ FAILED | Distributed training errors |
| 512 | 4096 | ✅ SUCCESS | Ran successfully, no OOM |

## Working Configuration

**Config File:** `configs/ssl_config_test_batch2_modified.py`

Key settings:
- `batch_size`: 512 (per GPU)
- `apply_qat_representation`: False (QAT disabled)
- `apply_amp`: False (AMP disabled)
- `cache_size`: 10 (reduced for faster startup)
- `num_workers`: 2
- `pin_memory`: True

## Command to Run

```bash
# Navigate to working directory
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml

# Run training with 8 GPUs
apptainer exec --nv ../ml_env.sif torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_ssl_multi_gpu.py \
    --config configs/ssl_config_test_batch2_modified.py
```

## Performance Metrics (at step 8)

- **Loss:** 24814.432
- **Global Batch Size:** 4096
- **Examples/sec:** 198.2
- **Learning Rate:** 0.00000 (warmup phase)
- **GPU Utilization:** No memory errors
- **Rank(z_proj):** 0.980
- **Rank(repr_f32):** 0.574

## Interactive Shell Setup

To get an interactive shell with the same GPU resources:

```bash
# Using sh_dev (simpler)
sh_dev -p serc -t 12:00:00 -c 32 -m 512GB -g 8

# Using srun (with GPU constraints)
srun -p serc -t 12:00:00 -c 32 --mem=512G --gres=gpu:8 \
     -C GPU_SKU:A100_SXM4,GPU_MEM:80GB --pty bash
```

## Model Architecture

- **S2 Encoder:** 4 layers, 4 heads, 4096 feedforward dim
- **S1 Encoder:** 4 layers, 4 heads, 4096 feedforward dim  
- **Projection Head:** 16,384 output dim, 16,384 hidden dim
- **Total Parameters:** 2,464,303,234
- **Latent Dim:** 128
- **Fusion Method:** concat

## Next Steps

1. **Longer Training:** Run with more steps/epochs to verify stability
2. **QAT Testing:** Try enabling QAT with `qat_representation_start_step: 5000`
3. **AMP Testing:** Try enabling AMP for potential memory savings
4. **Batch Size Optimization:** Test intermediate batch sizes (e.g., 384, 768)
5. **Multi-node Testing:** Scale to 2+ nodes for larger batch sizes

## Troubleshooting

- **OOM Errors:** Reduce batch size (try 256, 128, 64)
- **Distributed Errors:** Check NCCL settings and network connectivity
- **Container Issues:** Verify `../ml_env.sif` path exists

## Files Modified

- `configs/ssl_config_test_batch2_modified.py` - Working config with batch size 512
- `test_modified_batch2_4gpu.sbatch` - Sbatch script (last used for 16 GPUs across 2 nodes)