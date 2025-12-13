# Training Configurations

This directory contains configuration files for training the TESSERA SSL model with FSDP multi-GPU support.

## Available Configurations

### 1. `ssl_config_32k.py` - 32k Batch on 8x 40GB GPUs ⭐ **RECOMMENDED**

**Use for**: Production training with 32k effective batch size on 8x 40GB A100 GPUs

**Key features:**
- Batch size: 256 per GPU
- Gradient accumulation: 16 steps
- Effective batch: 256 × 8 × 16 = **32,768**
- FSDP memory sharding
- AMP enabled
- QAT enabled after warmup

**Estimated time**: 
- 8x 40GB A100 GPUs: ~6-8 hours per epoch
- Memory: ~18 GB per GPU (safe for 40GB)

**Command:**
```bash
sbatch train_multi_gpu.sbatch  # Uses this config by default
```

### 2. `ssl_config_2gpu.py` - 2 GPU Testing

**Use for**: Testing FSDP + gradient accumulation before scaling to 8 GPUs

**Key features:**
- Batch size: 256 per GPU
- Gradient accumulation: 8 steps
- Effective batch: 256 × 2 × 8 = **4,096**
- Same architecture as 32k config
- More frequent logging for debugging

**Estimated time**:
- 2x 40GB A100 GPUs: ~2-3 hours per epoch
- Memory: ~18 GB per GPU

**Command:**
```bash
sbatch train_2gpu.sbatch  # Uses this config by default
```

### 3. `ssl_config.py` - Full Production Training (Legacy)

**Use for**: Reference config, can be used for 80GB GPUs without gradient accumulation

**Key features:**
- Large batch size (1024 per GPU)
- No gradient accumulation
- Requires 80GB GPUs or memory optimizations

**Memory**: ~45 GB per GPU (requires 80GB A100s)

### 4. `ssl_config_minimal.py` - Quick Testing

**Use for**: Debugging, testing changes, or quick validation

**Key features:**
- Small batch size (32)
- Smaller model (2 layers instead of 4)
- Smaller projection head (2048 dims)
- Validation disabled
- No QAT
- No mixup

**Estimated time**:
- 1 GPU: ~5-10 minutes for few steps
- Memory: ~8-12 GB per GPU

**Recommended for:**
- Verifying data paths are correct
- Testing code changes
- Quick sanity checks
- Single GPU debugging

## How to Use

### Recommended Workflow

**Step 1: Test on 2 GPUs first**
```bash
sbatch train_2gpu.sbatch  # Uses ssl_config_2gpu.py by default
```

**Step 2: Scale to 8 GPUs for production**
```bash
sbatch train_multi_gpu.sbatch  # Uses ssl_config_32k.py by default
```

### Using Custom Configs

**Option 1: Specify config as argument**
```bash
sbatch train_multi_gpu.sbatch configs/ssl_config.py
sbatch train_2gpu.sbatch configs/my_custom_config.py
```

**Option 2: Create your own config**
```bash
# Copy and modify
cp configs/ssl_config_2gpu.py configs/my_config.py
vim configs/my_config.py

# Run with your config
sbatch train_2gpu.sbatch configs/my_config.py
```

## Critical Fields to Update

Before running, **you MUST update these paths** in your config:

```python
config = {
    # UPDATE THESE!
    "index_dir": "/path/to/your/parquet_files",
    "data_root": "/path/to/your/tile_data",
    
    # Optionally update validation paths if you want validation
    "val_s2_bands_file_path": "/path/to/validation/bands.npy",
    # ... other validation paths ...
}
```

## Field Reference

### Required Fields (for TreeDataset)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `index_dir` | str | Path to parquet index files | `"/scratch/.../parquet_files"` |
| `data_root` | str | Path to tile data directory | `"/scratch/.../tile_data"` |
| `years` | list[int] | Years to sample from | `[2017, 2018, 2019, 2020]` |
| `sample_size_s2` | int | S2 time steps | `40` |
| `sample_size_s1` | int | S1 time steps | `40` |
| `batch_size` | int | Batch size per GPU | `32` or `2048` |
| `epochs` | int | Number of epochs | `10` |
| `learning_rate` | float | Initial learning rate | `0.001` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_size` | int | 50 | LRU cache size for tiles |
| `num_workers` | int | 4 | DataLoader workers per GPU |
| `apply_amp` | bool | False | Mixed precision training |
| `apply_mixup` | bool | False | Mixup augmentation |
| `apply_qat_representation` | bool | False | Quantization aware training |
| `val_interval_steps` | int | 0 | Steps between validations (0=disabled) |
| `use_torch_compile` | bool | False | PyTorch 2.0+ compile |

### Resume Training Fields

| Field | Type | Description |
|-------|------|-------------|
| `resume_from_checkpoint` | str or None | Path to checkpoint to resume from |
| `resume_learning_rate` | float | LR for resumed training (optional) |
| `resume_warmup_steps` | int | Warmup steps after resume (optional) |

## Removed Fields (from old configs)

These fields are **NO LONGER NEEDED** with TreeDataset:

- ❌ `rust_cmd` - No data generation
- ❌ `chunk_batch` - No chunk-based loading
- ❌ `total_samples` - Calculated from dataset

If your old config has these, they will be ignored.

## Example: Customizing for Your Cluster

```python
config = {
    # Your cluster paths
    "index_dir": "/work/username/tessera/parquet_index",
    "data_root": "/work/username/tessera/tiles",
    
    # Adjust for your GPU memory
    "batch_size": 256,  # Smaller if you have < 40GB GPUs
    
    # Adjust for your training time
    "epochs": 100,
    
    # Disable validation if you don't have validation data
    "val_interval_steps": 0,
    
    # Enable AMP for faster training if stable
    "apply_amp": True,
    
    # Rest of config...
}
```

## Troubleshooting

### "index_dir not found"
Check that the path exists and contains `.parquet` files:
```bash
ls /path/to/index_dir/*.parquet
```

### "Dataset is empty"
Check that:
1. Years match available data
2. Parquet files have data for those years
3. Paths are correct

### Out of memory
Reduce these in order:
1. `batch_size` (try 1024, 512, 256, 128, 64, 32)
2. `cache_size` (try 25, 10)
3. `num_workers` (try 1, 0)

## Config Hierarchy

When running training:
1. Base config is loaded from the file
2. CLI arguments override config values (if any)
3. Some values are calculated dynamically (e.g., total_steps)

## Best Practices

1. ✅ **Test on 2 GPUs first** - Validate FSDP + gradient accumulation work
2. ✅ **Then scale to 8 GPUs** - Once 2 GPU run is successful
3. ✅ **Use absolute paths** for data directories
4. ✅ **Version control your configs** (use git)
5. ✅ **Comment your changes** in config files
6. ✅ **Keep separate configs** for different experiments
7. ⚠️ **Avoid single GPU with FSDP** - FSDP adds overhead without benefits on 1 GPU

## More Info

- See `TRAINING_GUIDE.md` for full training instructions
- See `IMPLEMENTATION_SUMMARY.md` for code changes
- See original `tessera-util/configs/ssl_config.py` for reference

