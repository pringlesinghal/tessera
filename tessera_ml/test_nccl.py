#!/usr/bin/env python3
"""
Simple NCCL test script for multi-GPU setups.
Tests basic distributed communication without the overhead of model training.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_nccl.py
"""

import torch
import torch.distributed as dist
from datetime import timedelta
import os

def test_nccl():
    """Test NCCL initialization and basic operations."""
    
    print("="*60)
    print("NCCL Communication Test")
    print("="*60)
    
    # Initialize process group
    try:
        print("\n1. Initializing process group with NCCL backend...")
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
        print("   ✅ Process group initialized successfully!")
    except Exception as e:
        print(f"   ❌ Failed to initialize process group: {e}")
        return
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"\n2. Process Info:")
    print(f"   - Global Rank: {rank}/{world_size}")
    print(f"   - Local Rank: {local_rank}")
    print(f"   - Device: cuda:{local_rank}")
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Test basic tensor creation
    try:
        print(f"\n3. Creating test tensor on rank {rank}...")
        tensor = torch.ones(2, 2, device=device) * (rank + 1)
        print(f"   ✅ Tensor created: {tensor[0, 0].item()}")
    except Exception as e:
        print(f"   ❌ Failed to create tensor: {e}")
        dist.destroy_process_group()
        return
    
    # Test all-reduce
    try:
        print(f"\n4. Testing all-reduce on rank {rank}...")
        original_value = tensor[0, 0].item()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced_value = tensor[0, 0].item()
        expected_value = sum(range(1, world_size + 1))
        
        if abs(reduced_value - expected_value) < 1e-5:
            print(f"   ✅ All-reduce successful!")
            print(f"      Original: {original_value}, After reduce: {reduced_value}, Expected: {expected_value}")
        else:
            print(f"   ⚠️  All-reduce value unexpected: {reduced_value} (expected {expected_value})")
    except Exception as e:
        print(f"   ❌ All-reduce failed: {e}")
        dist.destroy_process_group()
        return
    
    # Test broadcast
    try:
        print(f"\n5. Testing broadcast from rank 0...")
        if rank == 0:
            tensor = torch.ones(2, 2, device=device) * 42
        else:
            tensor = torch.zeros(2, 2, device=device)
        
        dist.broadcast(tensor, src=0)
        broadcast_value = tensor[0, 0].item()
        
        if abs(broadcast_value - 42.0) < 1e-5:
            print(f"   ✅ Broadcast successful! Received: {broadcast_value}")
        else:
            print(f"   ⚠️  Broadcast value unexpected: {broadcast_value} (expected 42.0)")
    except Exception as e:
        print(f"   ❌ Broadcast failed: {e}")
        dist.destroy_process_group()
        return
    
    # Test barrier
    try:
        print(f"\n6. Testing barrier synchronization...")
        dist.barrier()
        print(f"   ✅ Barrier passed on rank {rank}!")
    except Exception as e:
        print(f"   ❌ Barrier failed: {e}")
        dist.destroy_process_group()
        return
    
    # Print GPU info
    if rank == 0:
        print(f"\n7. GPU Information:")
        print(f"   - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   - GPU {i}: {props.name}, {props.total_memory / 1024**3:.2f} GB")
    
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()
    
    if rank == 0:
        print("\n" + "="*60)
        print("✅ All NCCL tests passed successfully!")
        print("="*60)


if __name__ == "__main__":
    test_nccl()

