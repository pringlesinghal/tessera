#!/usr/bin/env python3
"""
Direct memory test - bypasses dataset loading to test batch sizes immediately
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy

# Import your models
from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel, BarlowTwinsLoss

def test_batch_size(batch_size, enable_amp=True):
    """Test a specific batch size with synthetic data"""
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    print(f"Testing batch_size={batch_size}, AMP={enable_amp}")
    
    # Create model (same config as your working setup)
    s2_enc = TransformerEncoder(
        band_num=10, latent_dim=128, nhead=4, num_encoder_layers=4,
        dim_feedforward=4096, dropout=0.1, max_seq_len=40
    ).to(device)
    
    s1_enc = TransformerEncoder(
        band_num=2, latent_dim=128, nhead=4, num_encoder_layers=4,
        dim_feedforward=4096, dropout=0.1, max_seq_len=40
    ).to(device)
    
    projector = ProjectionHead(128, 4096, 4096).to(device)
    
    model = MultimodalBTModel(s2_enc, s1_enc, projector, fusion_method="concat", 
                             return_repr=True, latent_dim=128).to(device)
    
    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # FSDP with mixed precision
    mixed_precision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    ) if enable_amp else None
    
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": device,
        "sync_module_states": True,
    }
    if mixed_precision:
        fsdp_config["mixed_precision"] = mixed_precision
    
    model = FSDP(model, **fsdp_config)
    print(f"FSDP model created. Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Create synthetic batch
    s2_data = torch.randn(batch_size, 40, 10, device=device, dtype=torch.float32)
    s1_data = torch.randn(batch_size, 40, 2, device=device, dtype=torch.float32)
    
    print(f"Synthetic batch created: s2={s2_data.shape}, s1={s1_data.shape}")
    print(f"Memory after batch: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Test forward pass
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = BarlowTwinsLoss(lambda_coeff=0.005)
    
    try:
        # Forward pass
        with torch.cuda.amp.autocast(enabled=enable_amp):
            proj1, repr1 = model(s2_data, s1_data)
            proj2, repr2 = model(s2_data, s1_data)  # Same data for test
            loss, _, _ = criterion(proj1, proj2)
        
        print(f"Forward pass SUCCESS. Loss: {loss.item():.6f}")
        print(f"Memory after forward: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Backward pass
        optimizer.zero_grad()
        scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Backward pass SUCCESS")
        print(f"Memory after backward: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        return True, torch.cuda.memory_allocated()/1024**3
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM ERROR: {e}")
            return False, torch.cuda.memory_allocated()/1024**3
        else:
            raise e
    
    finally:
        torch.cuda.empty_cache()

def main():
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("🧪 Direct Memory Test - 4 A100 GPUs")
        print("Testing batch sizes without dataset loading")
    
    # Test sequence
    test_cases = [
        (2, False),     # Your working baseline
        (64, True),     # Small with AMP  
        (256, True),    # Medium
        (512, True),    # Large  
        (1024, True),   # Target: 4096 global
        (2048, True),   # Stretch: 8192 global
    ]
    
    results = []
    
    for batch_size, amp in test_cases:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"\n{'='*50}")
            print(f"Testing batch_size={batch_size} (global={batch_size*4})")
        
        success, peak_memory = test_batch_size(batch_size, amp)
        
        result = {
            'batch_size': batch_size,
            'global_batch_size': batch_size * 4,
            'amp': amp,
            'success': success,
            'peak_memory_gb': peak_memory
        }
        results.append(result)
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            status = "✅" if success else "❌"
            print(f"Result: {status} Peak memory: {peak_memory:.2f} GB")
        
        # If we OOM, don't test larger batches
        if not success and batch_size >= 256:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print("Stopping due to OOM...")
            break
    
    # Summary
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n{'='*60}")
        print("MEMORY TEST RESULTS")
        print(f"{'='*60}")
        print(f"{'Batch':<8} {'Global':<8} {'AMP':<6} {'Memory':<10} {'Status':<8}")
        print(f"{'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*8}")
        
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"{r['batch_size']:<8} {r['global_batch_size']:<8} {r['amp']!s:<6} "
                  f"{r['peak_memory_gb']:<10.2f} {status:<8}")
        
        # Find optimal
        successful = [r for r in results if r['success']]
        target_reached = [r for r in successful if r['global_batch_size'] >= 4096]
        
        if target_reached:
            best = max(target_reached, key=lambda x: x['global_batch_size'])
            print(f"\n🎯 TARGET REACHED!")
            print(f"   Optimal: batch_size={best['batch_size']} (global={best['global_batch_size']})")
            print(f"   Memory usage: {best['peak_memory_gb']:.2f} GB per GPU")
        elif successful:
            best = max(successful, key=lambda x: x['global_batch_size'])
            print(f"\n⚠️ Target not reached. Best:")
            print(f"   Max: batch_size={best['batch_size']} (global={best['global_batch_size']})")
            print(f"   Memory usage: {best['peak_memory_gb']:.2f} GB per GPU")
        else:
            print("\n❌ No successful configurations!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()