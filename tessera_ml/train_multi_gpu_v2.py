#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-GPU FSDP Training Script - Adapted from tessera-util reference implementation
Incremental approach: Start with basic distributed training, then add FSDP step by step

Usage:
  torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \
    tessera_ml/train_multi_gpu_v2.py --config config.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

from datetime import timedelta
import math
import time
import gc
import argparse
import logging
import subprocess
from datetime import datetime
from contextlib import nullcontext
import json
import copy

import numpy as np
import torch
torch.set_float32_matmul_precision('high') 

import torch.amp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# FSDP imports - we'll add these incrementally
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType
)

import wandb

# Our working imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tessera_ml.dataset import TreeDataset
from tessera_ml.models.modules import TransformerEncoder, ProjectionHead
from tessera_ml.models.ssl_model import MultimodalBTModel, BarlowTwinsLoss, compute_cross_correlation
from tessera_ml.utils.lr_scheduler import adjust_learning_rate
from tessera_ml.utils.metrics import linear_probe_evaluate, rankme
from tessera_ml.utils.misc import remove_dir, save_checkpoint, plot_cross_corr

# ============== UTILITY FUNCTIONS ==============

def get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

def print_model_info(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_model_structure(m, depth=0, max_depth=20):
        if depth > max_depth: return "  " * depth + "...\n"
        res = ""
        for n, child in m.named_children():
            res += "  " * depth + f"({n}): {child.__class__.__name__}\n"
            res += get_model_structure(child, depth + 1, max_depth)
        return res
    
    model_structure = get_model_structure(model)
    return (f"\n{name} Information:\n"
            f"Model Class: {model.__class__.__name__}\n"
            f"Total Parameters: {total_params:,}\n"
            f"Trainable Parameters: {trainable_params:,}\n"
            f"Model Structure:\n{model_structure}")

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU FSDP Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--test_only', action='store_true', help='Only test distributed initialization')
    return parser.parse_args()

# ============== MAIN FUNCTION ==============

def main():
    # Initialize seeds for reproducibility
    torch.manual_seed(3407)
    np.random.seed(3407)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Configure NCCL for container environments BEFORE distributed init
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^docker0,lo")
    os.environ.setdefault("NCCL_DEBUG", "INFO") 
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    # === DISTRIBUTED INITIALIZATION ===
    print("Initializing distributed process group...")
    try:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60)) 
        print("✅ NCCL initialization successful")
    except Exception as e:
        print(f"❌ NCCL initialization failed: {e}")
        print("Trying gloo backend as fallback...")
        try:
            dist.init_process_group(backend='gloo', timeout=timedelta(minutes=60))
            print("✅ Gloo initialization successful")
        except Exception as e2:
            print(f"❌ Both NCCL and gloo failed: {e2}")
            raise e2
    
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    nnodes = world_size // nproc_per_node if nproc_per_node > 0 else world_size

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # === LOGGING SETUP ===
    if global_rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        if not os.environ.get("WANDB_MODE", "") == "offline": 
            os.environ['WANDB_MODE'] = 'disabled'

    # Parse arguments and load config
    args_cli = parse_args()
    
    # Test mode: just verify distributed setup works
    if args_cli.test_only:
        if global_rank == 0:
            logging.info("Running in test-only mode: verifying distributed setup")
        
        # Test basic tensor operations
        test_tensor = torch.tensor([global_rank], dtype=torch.float32).to(device)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        if global_rank == 0:
            expected = sum(range(world_size))
            logging.info(f"All-reduce test: got {test_tensor.item()}, expected {expected}")
            if abs(test_tensor.item() - expected) < 1e-6:
                logging.info("✅ Distributed communication test PASSED")
            else:
                logging.error("❌ Distributed communication test FAILED")
        
        dist.destroy_process_group()
        return

    # Load configuration
    config_module = {}
    with open(args_cli.config, "r") as f: 
        exec(f.read(), config_module)
    config = config_module['config']

    if global_rank == 0:
        logging.info(f"Running on {world_size} GPU(s) across {nnodes} node(s). GlobalRank={global_rank}, LocalRank={local_rank}, device={device}")
        logging.info(f"Initial GPU memory usage: {get_gpu_memory_usage():.2f} MB")
    
    # === DATASET SETUP (adapted from working single-GPU version) ===
    if global_rank == 0:
        logging.info("Initializing TreeDataset...")
    
    dataset_train = TreeDataset(
        index_dir=config["index_dir"],
        data_dir=config["data_root"],
        year=None,
        years=config["years"],
        sample_size_s2=config["sample_size_s2"],
        sample_size_s1=config["sample_size_s1"],
        normalize=config.get("normalize", True),
        cache_size=config.get("cache_size", 100),
    )

    # Create distributed sampler
    train_sampler = DistributedSampler(
        dataset_train, 
        num_replicas=world_size, 
        rank=global_rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset_train,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=config.get("num_workers", 2),
        pin_memory=config.get("pin_memory", True),
        drop_last=True
    )

    if global_rank == 0:
        logging.info(f"Dataset size: {len(dataset_train)}")
        logging.info(f"DataLoader batches per epoch: {len(dataloader)}")

    # === MODEL SETUP (from working single-GPU version) ===
    s2_enc = TransformerEncoder(
        band_num=10,
        latent_dim=config["latent_dim"],
        nhead=config.get('s2_num_heads', 4),
        num_encoder_layers=config.get('s2_num_layers', 4),
        dim_feedforward=config.get('s2_dim_feedforward', 1024),
        dropout=0.1,
        max_seq_len=config["sample_size_s2"],
    ).to(device)

    s1_enc = TransformerEncoder(
        band_num=2,
        latent_dim=config["latent_dim"],
        nhead=config.get('s1_num_heads', 4),
        num_encoder_layers=config.get('s1_num_layers', 4),
        dim_feedforward=config.get('s1_dim_feedforward', 1024),
        dropout=0.1,
        max_seq_len=config["sample_size_s1"],
    ).to(device)

    projector = ProjectionHead(
        config["latent_dim"], 
        config["projector_hidden_dim"], 
        config["projector_out_dim"]
    ).to(device)

    model = MultimodalBTModel(
        s2_enc, s1_enc, projector,
        fusion_method=config["fusion_method"],
        return_repr=True, 
        latent_dim=config["latent_dim"],
        apply_qat_representation=config.get('apply_qat_representation', False),
        qat_representation_bits=config.get('qat_representation_bits', 8)
    ).to(device)

    if global_rank == 0:
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"GPU memory after model: {get_gpu_memory_usage():.2f} MB")

    # === START WITH BASIC DDP (not FSDP yet) ===
    # For now, let's use standard DDP to verify everything works
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    if global_rank == 0:
        logging.info("Wrapping model with DDP (testing phase)")
    
    model = DDP(model, device_ids=[local_rank])
    
    # === OPTIMIZER SETUP ===
    criterion = BarlowTwinsLoss(lambda_coeff=config['barlow_lambda'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # === POST-NORMALIZATION SETUP ===
    # Sentinel-2 normalization parameters (from tessera codebase)
    s2_mean = torch.tensor([
        1353.3418, 1265.4015, 1269.009, 1976.1317,
        2581.0518, 3238.1504, 3546.2078, 3459.1382,
        3906.4131, 1535.5925
    ], device=device, dtype=torch.float32).view(1, 1, -1)  # [1, 1, 10]
    
    s2_std = torch.tensor([
        242.07303, 290.84450, 402.9476, 516.77747,
        729.89056, 928.1345, 896.01056, 896.4886,
        1111.1157, 234.4575
    ], device=device, dtype=torch.float32).view(1, 1, -1)  # [1, 1, 10]
    
    # Sentinel-1 normalization parameters
    s1_mean = torch.tensor([-9.205532, -16.072447], device=device, dtype=torch.float32).view(1, 1, -1)  # [1, 1, 2]
    s1_std = torch.tensor([2.3895948, 4.153361], device=device, dtype=torch.float32).view(1, 1, -1)  # [1, 1, 2]
    
    def normalize_batch_on_gpu(s2_raw, s1_raw):
        """Apply post-normalization on GPU for efficiency"""
        if config.get("normalize", True):
            # Already normalized in dataset
            return s2_raw, s1_raw
        
        # Extract bands (exclude DOY which is last channel)
        s2_bands = s2_raw[:, :, :-1]  # [batch, time, 10]
        s2_doy = s2_raw[:, :, -1:]    # [batch, time, 1] 
        
        s1_bands = s1_raw[:, :, :-1]  # [batch, time, 2]
        s1_doy = s1_raw[:, :, -1:]    # [batch, time, 1]
        
        # Normalize bands
        s2_norm = (s2_bands - s2_mean) / s2_std
        s1_norm = (s1_bands - s1_mean) / s1_std
        
        # Concatenate back with DOY
        s2_final = torch.cat([s2_norm, s2_doy], dim=-1)
        s1_final = torch.cat([s1_norm, s1_doy], dim=-1)
        
        return s2_final, s1_final
    
    if global_rank == 0:
        logging.info("Starting training loop...")

    # === TRAINING LOOP WITH POST-NORMALIZATION ===
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Mixed precision setup
    use_amp = config.get('apply_amp', True)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to GPU
            s2_raw = batch['s2_aug1'].to(device, non_blocking=True)
            s1_raw = batch['s1_aug1'].to(device, non_blocking=True)
            
            # Apply post-normalization on GPU
            s2_data, s1_data = normalize_batch_on_gpu(s2_raw, s1_raw)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    z_s2, z_s1, repr_out = model(s2_data, s1_data)
                    loss = criterion(z_s2, z_s1)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.get('clip_grad_norm', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                z_s2, z_s1, repr_out = model(s2_data, s1_data)
                loss = criterion(z_s2, z_s1)
                
                loss.backward()
                
                if config.get('clip_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if global_rank == 0 and batch_idx % config.get('log_interval_steps', 10) == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, GPU Mem = {get_gpu_memory_usage():.1f}MB")
            
            # Learning rate scheduling
            if batch_idx % 100 == 0:
                adjust_learning_rate(
                    optimizer, 
                    epoch * len(dataloader) + batch_idx,
                    config['learning_rate'],
                    warmup_steps=int(config.get('warmup_ratio', 0.1) * len(dataloader)),
                    total_steps=config['epochs'] * len(dataloader)
                )

    if global_rank == 0:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logging.info(f"✅ Training test completed! Average loss: {avg_loss:.4f}")

    # Cleanup
    dist.destroy_process_group()
    
    if global_rank == 0:
        logging.info("Multi-GPU training test successful!")

if __name__ == "__main__":
    main()