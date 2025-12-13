#!/usr/bin/env python3
"""
Minimal distributed test to verify NCCL communication works
Run with: torchrun --nproc_per_node=2 tessera_ml/test_dist_minimal.py
"""

import os
import torch
import torch.distributed as dist
from datetime import timedelta

def test_basic_distributed():
    """Test basic distributed setup without any models"""
    print("Starting basic distributed test...")
    
    # Configure NCCL for container environments
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^docker0,lo")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    
    try:
        # Initialize process group
        print("Initializing process group...")
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=10))
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        
        print(f"Rank {rank}/{world_size}, Local Rank {local_rank}, Device: {device}")
        
        # Test basic communication
        tensor = torch.tensor([rank], dtype=torch.float32).to(device)
        print(f"Rank {rank}: Before all_reduce, tensor = {tensor}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank}: After all_reduce, tensor = {tensor}")
        
        # Test broadcast
        if rank == 0:
            broadcast_tensor = torch.tensor([42.0]).to(device)
        else:
            broadcast_tensor = torch.zeros(1).to(device)
            
        dist.broadcast(broadcast_tensor, src=0)
        print(f"Rank {rank}: After broadcast, tensor = {broadcast_tensor}")
        
        print(f"Rank {rank}: Basic distributed test PASSED!")
        
        # Cleanup
        dist.destroy_process_group()
        return True
        
    except Exception as e:
        print(f"Rank {rank if 'rank' in locals() else '?'}: Basic distributed test FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_distributed()
    if success:
        print("✅ Distributed communication test successful")
    else:
        print("❌ Distributed communication test failed")
        exit(1)