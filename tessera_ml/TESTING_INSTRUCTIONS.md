# Multi-GPU Training Testing Instructions

## Overview
This document provides step-by-step instructions for testing multi-GPU training on Sherlock using containers and proper partition allocation.

## Prerequisites
- SSH access to Sherlock
- Access to dlobell group scratch space
- Container `ml_env.sif` built and ready

## Testing Phases

### Phase 0: Container Verification (2-3 minutes)
**Purpose:** Verify container has all dependencies

```bash
cd /scratch/groups/dlobell/psinghal/sentineldownloader/tessera
sbatch tessera_ml/verify_container.sbatch
```

**Expected outcome:** Container has PyTorch, torchrun, and all our modules
**If fails:** Rebuild container or check dependencies

### Phase 1: Distributed Communication Test (5-10 minutes)
**Purpose:** Test basic NCCL communication on dev partition

```bash
sbatch tessera_ml/test_dist_dev.sbatch
```

**Expected outcome:** 
- NCCL initialization successful
- Basic tensor operations work across GPUs
- Training script distributed init works

**If fails:** 
- Check NCCL configuration
- Try gloo backend fallback
- Verify GPU access in container

### Phase 2: DDP Training Test - Dev Partition (10-15 minutes)
**Purpose:** Test actual training with DDP on lightweight GPUs

```bash
sbatch tessera_ml/test_ddp_dev.sbatch
```

**Expected outcome:**
- Model loads successfully
- Dataset integration works
- Training loop completes without errors
- Memory usage reasonable

**If fails:**
- Reduce batch size in `test_config.py`
- Check dataset path accessibility
- Verify model fits in GPU memory

### Phase 3: DDP Training Test - A100s (15-20 minutes)
**Purpose:** Scale up to production hardware

```bash
sbatch tessera_ml/test_ddp_serc.sbatch
```

**Expected outcome:**
- Larger batch size (256) works
- Better performance on A100s
- Full model parameters fit
- Training completes successfully

**If fails:**
- Check A100 memory limits
- Verify dataset access from serc partition
- Adjust batch size if needed

### Phase 4: FSDP Integration Test (20-30 minutes)
**Purpose:** Add FSDP for memory efficiency and larger models

```bash
# First add FSDP to the working DDP script
python tessera_ml/add_fsdp.py

# Then test FSDP
sbatch tessera_ml/test_fsdp_serc.sbatch
```

**Expected outcome:**
- FSDP initialization successful
- Model sharding works correctly
- Memory usage improved vs DDP
- Training convergence maintained

**If fails:**
- Revert to DDP version (backup exists)
- Check FSDP configuration parameters
- Try simpler auto-wrap policy

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output (replace JOBID)
tail -f tessera_ml/logs/test_*_JOBID.out

# Cancel job if needed
scancel JOBID
```

## Success Criteria

### Phase 0 ✅
- [ ] Container loads successfully
- [ ] PyTorch detects GPUs
- [ ] torchrun command available
- [ ] All modules import correctly

### Phase 1 ✅
- [ ] NCCL initialization works
- [ ] Basic distributed communication
- [ ] Training script starts without errors

### Phase 2 ✅
- [ ] DDP wrapper successful
- [ ] Dataset loading works
- [ ] Training loop completes
- [ ] Loss decreases over batches

### Phase 3 ✅
- [ ] A100 performance improvement
- [ ] Larger batch size works
- [ ] Full model architecture supported

### Phase 4 ✅
- [ ] FSDP initialization successful
- [ ] Memory efficiency gains
- [ ] Training quality maintained
- [ ] Ready for production scaling

## Troubleshooting

### Common Issues

**NCCL Errors:**
- Check `NCCL_*` environment variables
- Try gloo backend as fallback
- Verify GPU interconnect

**Memory Issues:**
- Reduce batch size
- Check model parameter count
- Monitor GPU memory usage

**Dataset Issues:**
- Verify path accessibility from compute nodes
- Check file permissions
- Test dataset loading separately

**Container Issues:**
- Rebuild with updated dependencies
- Check Apptainer version compatibility
- Verify GPU passthrough (`--nv` flag)

## Next Steps After Success

1. **Production Configuration:** Modify `production_config_template.py`
2. **Scale Up:** Test multi-node if needed
3. **Full Training:** Submit production job
4. **Monitoring:** Set up W&B monitoring

## File Structure

```
tessera_ml/
├── train_multi_gpu_v2.py       # Main training script
├── test_dist_minimal.py        # Minimal communication test
├── add_fsdp.py                 # FSDP integration tool
├── test_config.py              # Dev partition config
├── verify_container.sbatch     # Container verification
├── test_dist_dev.sbatch        # Dev communication test
├── test_ddp_dev.sbatch         # Dev DDP training
├── test_ddp_serc.sbatch        # A100 DDP training
├── test_fsdp_serc.sbatch       # A100 FSDP training
└── logs/                       # Job output logs
```