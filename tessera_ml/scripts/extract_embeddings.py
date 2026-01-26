#!/usr/bin/env python3
"""
Extract embeddings from all samples in the dataset using a pretrained checkpoint.

This script:
1. Loads a pretrained MultimodalBTModel checkpoint
2. Iterates through the entire TreeDataset
3. Extracts 128-dim embeddings (fused_representation_f32)
4. Saves embeddings.npy (memory-mapped) and index_mapping.npy for later use in clustering

Supports:
- Memory-mapped storage for 200M+ samples (~100GB embeddings)
- Multi-GPU processing with DDP (one process per GPU via torchrun)
- Checkpointing/resume for long-running jobs
- Auto-detection of model architecture (base model vs trained model)

Architecture Notes:
==================
There are TWO different model architectures to be aware of:

1. BASE MODEL (tessera_best_model_fsdp_20250427_084307.pt):
   - Uses standard nn.GRU in TemporalAwarePooling
   - Has 8 transformer layers
   - No layer_norm in attention pooling
   - Keys: temporal_context.weight_hh_l0, temporal_context.bias_ih_l0, etc.

2. TRAINED MODEL (newer checkpoints from tessera_ml training):
   - Uses CustomGRU in CustomTemporalAwarePooling
   - Has 4 transformer layers (configurable)
   - Has layer_norm in attention pooling
   - Keys: temporal_context.gru_cell.W_hh.weight, etc.

This script auto-detects the architecture from checkpoint keys.

Usage:
    torchrun --standalone --nproc_per_node=8 scripts/extract_embeddings.py \
        --checkpoint /path/to/checkpoint.pt \
        --output_dir /path/to/output \
        --batch_size 512 \
        --num_workers 2
"""

import argparse
import os
import sys
import logging
import time
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TreeDataset
from models.modules import (
    TemporalPositionalEncoder,
    TemporalAwarePooling,
    CustomTemporalAwarePooling,
    ProjectionHead
)
from models.ssl_model import MultimodalBTModel

# Logger configured in main() based on rank
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from pretrained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="curation_outputs",
        help="Directory to save embeddings and index mapping"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid",
        help="Path to the parquet index directory"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/users/psinghal/time_series",
        help="Root directory for time series data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Per-GPU batch size for embedding extraction"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers per process"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Save progress every N batches"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to extract (default: all samples). Use 50000000 for 50M."
    )
    return parser.parse_args()


def detect_architecture(state_dict):
    """
    Detect model architecture from checkpoint keys.

    Returns dict with:
        - use_custom_gru: bool (True for trained model, False for base model)
        - num_layers: int (typically 8 for base, 4 for trained)
        - has_layer_norm: bool
        - model_type: str ("base_model" or "trained_model")
    """
    keys = list(state_dict.keys())

    # Detect GRU type by looking at key patterns
    use_custom_gru = any("gru_cell.W_" in k for k in keys)
    has_standard_gru = any("temporal_context.weight_hh_l0" in k for k in keys)

    if use_custom_gru:
        gru_type = "CustomGRU"
        model_type = "trained_model"
    elif has_standard_gru:
        gru_type = "Standard nn.GRU"
        model_type = "base_model"
    else:
        gru_type = "Unknown"
        model_type = "unknown"

    # Detect layer_norm presence
    has_layer_norm = any("attn_pool.layer_norm" in k for k in keys)

    # Count transformer layers
    layer_indices = set()
    for k in keys:
        if "transformer_encoder.layers." in k:
            parts = k.split("transformer_encoder.layers.")
            if len(parts) > 1:
                layer_idx = parts[1].split(".")[0]
                if layer_idx.isdigit():
                    layer_indices.add(int(layer_idx))

    num_layers = max(layer_indices) + 1 if layer_indices else 4

    arch_info = {
        "use_custom_gru": use_custom_gru,
        "has_layer_norm": has_layer_norm,
        "num_layers": num_layers,
        "model_type": model_type,
        "gru_type": gru_type,
    }

    return arch_info


def infer_config_from_state_dict(state_dict):
    """Infer model configuration from checkpoint state dict."""
    config = {
        "latent_dim": 128,
        "fusion_method": "concat",
        "sample_size_s2": 40,
        "sample_size_s1": 40,
    }

    # Detect architecture first
    arch_info = detect_architecture(state_dict)
    config["use_custom_gru"] = arch_info["use_custom_gru"]
    config["model_type"] = arch_info["model_type"]

    # Count transformer layers by looking at layer indices
    s2_layers = set()
    s1_layers = set()
    for key in state_dict.keys():
        if "s2_backbone.transformer_encoder.layers." in key:
            layer_num = int(key.split(".")[3])
            s2_layers.add(layer_num)
        if "s1_backbone.transformer_encoder.layers." in key:
            layer_num = int(key.split(".")[3])
            s1_layers.add(layer_num)

    config["s2_num_layers"] = len(s2_layers) if s2_layers else 4
    config["s1_num_layers"] = len(s1_layers) if s1_layers else 4

    # Infer nhead from in_proj_weight shape
    for key, value in state_dict.items():
        if "s2_backbone.transformer_encoder.layers.0.self_attn.in_proj_weight" in key:
            d_model = value.shape[1]
            config["latent_dim"] = d_model // 4
            config["s2_num_heads"] = 8 if d_model >= 512 else 4
            config["s1_num_heads"] = config["s2_num_heads"]
            break

    # Infer dim_feedforward from linear1 weight
    for key, value in state_dict.items():
        if "s2_backbone.transformer_encoder.layers.0.linear1.weight" in key:
            config["s2_dim_feedforward"] = value.shape[0]
            config["s1_dim_feedforward"] = value.shape[0]
            break

    # Infer projector structure from checkpoint
    # Find all projector layer indices that have 2D weight tensors (Linear layers)
    projector_linear_indices = sorted([
        int(key.split(".")[2]) for key, value in state_dict.items()
        if key.startswith("projector.net.") and key.endswith(".weight")
        and len(value.shape) == 2  # Only 2D tensors are Linear weights
    ])

    if projector_linear_indices:
        # Get first layer input dim
        for key, value in state_dict.items():
            if "projector.net.0.weight" in key and len(value.shape) == 2:
                config["projector_hidden_dim"] = value.shape[0]
                config["projector_input_dim"] = value.shape[1]
                break

        # Get last layer output dim
        last_layer_idx = projector_linear_indices[-1]
        for key, value in state_dict.items():
            if f"projector.net.{last_layer_idx}.weight" in key and len(value.shape) == 2:
                config["projector_out_dim"] = value.shape[0]
                break

        # Count total linear layers and compute hidden layers
        # Structure: input(1) + hidden(N) + output(1) = N+2 total linear layers
        num_linear_layers = len(projector_linear_indices)
        config["projector_num_hidden"] = max(0, num_linear_layers - 2)  # Subtract input and output
        config["projector_num_layers"] = num_linear_layers

        logger.info(f"Detected projector: {num_linear_layers} linear layers "
                    f"({config['projector_num_hidden']} hidden), "
                    f"hidden_dim={config.get('projector_hidden_dim')}, "
                    f"out_dim={config.get('projector_out_dim')}")

    return config


class ProjectionHeadUniversal(nn.Module):
    """
    Universal ProjectionHead that matches the checkpoint structure.

    The base model projector has structure:
        Linear(128, 16384) -> BN -> ReLU
        Linear(16384, 16384) -> BN -> ReLU  (repeated num_hidden times)
        Linear(16384, 16384)  # final output

    Total: 1 input + num_hidden + 1 output = num_hidden + 2 linear layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=4):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=False))

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=False))

        # Output layer (no BN/ReLU after)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerEncoderUniversal(nn.Module):
    """
    Universal TransformerEncoder that supports both GRU types.

    This allows loading both base model checkpoints (with nn.GRU) and
    trained model checkpoints (with CustomGRU).
    """
    def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=8,
                 dim_feedforward=4096, dropout=0.1, use_custom_gru=False):
        super().__init__()
        input_dim = band_num

        # Embedding to increase dimension
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4)
        )

        # Temporal Encoder for DOY as position encoding
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Choose pooling based on model type
        if use_custom_gru:
            self.attn_pool = CustomTemporalAwarePooling(latent_dim * 4)
        else:
            self.attn_pool = TemporalAwarePooling(latent_dim * 4)

    def forward(self, x):
        # x: (B, seq_len, bands + doy)
        bands = x[:, :, :-1]  # All columns except last one
        doy = x[:, :, -1]     # Last column is DOY

        # Embedding of bands
        bands_embedded = self.embedding(bands)  # (B, seq_len, latent_dim*4)
        temporal_encoding = self.temporal_encoder(doy)

        # Add temporal encoding to embedded bands
        x = bands_embedded + temporal_encoding
        x = self.transformer_encoder(x)
        x = self.attn_pool(x)
        return x


def create_model(config, device):
    """Create the model architecture matching the checkpoint."""
    latent_dim = config.get("latent_dim", 128)
    use_custom_gru = config.get("use_custom_gru", False)
    model_type = config.get("model_type", "unknown")

    logger.info(f"Creating model with architecture: {model_type}")
    logger.info(f"  - use_custom_gru: {use_custom_gru}")
    logger.info(f"  - latent_dim: {latent_dim}")
    logger.info(f"  - s2_num_layers: {config.get('s2_num_layers', 8)}")
    logger.info(f"  - s1_num_layers: {config.get('s1_num_layers', 8)}")

    # S2 Encoder with correct GRU type
    s2_enc = TransformerEncoderUniversal(
        band_num=10,
        latent_dim=latent_dim,
        nhead=config.get("s2_num_heads", 8),
        num_encoder_layers=config.get("s2_num_layers", 8),
        dim_feedforward=config.get("s2_dim_feedforward", 4096),
        use_custom_gru=use_custom_gru,
    ).to(device)

    # S1 Encoder with correct GRU type
    s1_enc = TransformerEncoderUniversal(
        band_num=2,
        latent_dim=latent_dim,
        nhead=config.get("s1_num_heads", 8),
        num_encoder_layers=config.get("s1_num_layers", 8),
        dim_feedforward=config.get("s1_dim_feedforward", 4096),
        use_custom_gru=use_custom_gru,
    ).to(device)

    # Projector - use Universal version that matches checkpoint structure
    # Note: Not used for embedding extraction but needed for loading checkpoint
    projector = ProjectionHeadUniversal(
        input_dim=latent_dim,
        hidden_dim=config.get("projector_hidden_dim", 16384),
        output_dim=config.get("projector_out_dim", 16384),
        num_hidden_layers=config.get("projector_num_hidden", 4),
    ).to(device)
    logger.info(f"  - projector: input={latent_dim}, hidden={config.get('projector_hidden_dim', 16384)}, "
                f"out={config.get('projector_out_dim', 16384)}, num_hidden={config.get('projector_num_hidden', 4)}")

    # Create model with return_repr=True to get embeddings
    model = MultimodalBTModel(
        s2_enc,
        s1_enc,
        projector,
        fusion_method=config.get("fusion_method", "concat"),
        return_repr=True,
        latent_dim=latent_dim,
        apply_qat_representation=False,
    ).to(device)

    return model


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint with FSDP-compatible key handling and config inference.

    This function auto-detects the model architecture from checkpoint keys
    and creates a matching model. It requires strict loading to succeed -
    if there's an architecture mismatch, it will fail with a clear error.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Handle FSDP-wrapped keys
    new_state_dict = {}
    key_prefix = ""
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("_fsdp_wrapped_module."):
            new_key = key[len("_fsdp_wrapped_module."):]
            key_prefix = "_fsdp_wrapped_module."
        elif key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
            key_prefix = "_orig_mod."
        new_state_dict[new_key] = value

    if key_prefix:
        logger.info(f"Removed key prefix: '{key_prefix}'")

    # Detect architecture from state dict
    arch_info = detect_architecture(new_state_dict)
    logger.info("=" * 50)
    logger.info("DETECTED MODEL ARCHITECTURE:")
    logger.info(f"  Model type: {arch_info['model_type']}")
    logger.info(f"  GRU type: {arch_info['gru_type']}")
    logger.info(f"  Transformer layers: {arch_info['num_layers']}")
    logger.info(f"  Has layer_norm in attn_pool: {arch_info['has_layer_norm']}")
    logger.info("=" * 50)

    # Get config from checkpoint or infer from state dict
    config = checkpoint.get("config", {})
    if not config:
        logger.info("No config in checkpoint, inferring from state dict...")
        config = infer_config_from_state_dict(new_state_dict)
        logger.info(f"Inferred config: {config}")
    else:
        # Even if config exists, ensure architecture info is set
        config["use_custom_gru"] = arch_info["use_custom_gru"]
        config["model_type"] = arch_info["model_type"]

    # Create model with inferred config
    model = create_model(config, device)

    # Load state dict with strict=True to verify architecture match
    try:
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("Successfully loaded checkpoint with strict=True")
    except RuntimeError as e:
        error_msg = str(e)
        logger.error("=" * 50)
        logger.error("STRICT LOADING FAILED - ARCHITECTURE MISMATCH!")
        logger.error("=" * 50)
        logger.error(f"Error: {error_msg[:500]}...")

        # Provide diagnostic information
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(new_state_dict.keys())

        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys

        if missing_keys:
            logger.error(f"\nMissing keys in checkpoint ({len(missing_keys)}):")
            for k in sorted(list(missing_keys))[:10]:
                logger.error(f"  - {k}")
            if len(missing_keys) > 10:
                logger.error(f"  ... and {len(missing_keys) - 10} more")

        if unexpected_keys:
            logger.error(f"\nUnexpected keys in checkpoint ({len(unexpected_keys)}):")
            for k in sorted(list(unexpected_keys))[:10]:
                logger.error(f"  - {k}")
            if len(unexpected_keys) > 10:
                logger.error(f"  ... and {len(unexpected_keys) - 10} more")

        raise RuntimeError(
            f"Architecture mismatch: checkpoint uses {arch_info['gru_type']} with {arch_info['num_layers']} layers. "
            f"Missing {len(missing_keys)} keys, {len(unexpected_keys)} unexpected keys."
        )

    model.eval()
    logger.info(f"Loaded checkpoint. Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def main():
    args = parse_args()

    # =============================================
    # DDP Initialization
    # =============================================
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Configure logging: only rank 0 logs INFO, others WARNING
    log_level = logging.INFO if global_rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f'%(asctime)s - Rank {global_rank} - %(levelname)s - %(message)s'
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    if global_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    logger.info(f"DDP initialized: {world_size} processes")
    logger.info(f"  Rank {global_rank}, device cuda:{local_rank}: {torch.cuda.get_device_name(local_rank)}")

    # =============================================
    # Load model (each rank loads independently onto its own GPU)
    # No DDP wrapping needed - inference only, no gradient sync
    # =============================================
    model, config = load_checkpoint(args.checkpoint, device)

    # =============================================
    # Create dataset and partition across ranks
    # =============================================
    logger.info(f"Creating dataset from {args.index_dir}")
    dataset = TreeDataset(
        index_dir=args.index_dir,
        data_dir=args.data_root,
        year=None,  # Random year per sample
        years=list(range(2016, 2025)),
        cache_size=100,  # Per-process cache (effective total: 100 * world_size)
        sample_size_s2=40,
        sample_size_s1=40,
        normalize=True,
    )

    full_dataset_size = len(dataset)
    embedding_dim = config.get("latent_dim", 128)
    logger.info(f"Full dataset size: {full_dataset_size:,} samples")

    # Limit samples if max_samples is specified
    total_samples = full_dataset_size
    if args.max_samples is not None and args.max_samples < total_samples:
        total_samples = args.max_samples
        logger.info(f"Limiting to {total_samples:,} samples (--max_samples)")

    logger.info(f"Processing {total_samples:,} samples across {world_size} GPUs")
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Partition into contiguous chunks per rank
    chunk_size = math.ceil(total_samples / world_size)
    rank_start = global_rank * chunk_size
    rank_end = min(rank_start + chunk_size, total_samples)
    rank_total = rank_end - rank_start

    logger.info(f"Rank {global_rank}: samples [{rank_start:,}, {rank_end:,}) = {rank_total:,} samples")

    # Create subset for this rank
    rank_indices = list(range(rank_start, rank_end))
    rank_dataset = Subset(dataset, rank_indices)

    # =============================================
    # Setup shared memory-mapped embeddings file
    # =============================================
    embeddings_path = output_dir / "embeddings.npy"
    embeddings_size_gb = (total_samples * embedding_dim * 4) / (1024**3)

    # Rank 0 creates the memmap file, others wait
    if global_rank == 0:
        if embeddings_path.exists() and args.resume:
            logger.info("Existing memmap file found (resume mode)")
        else:
            logger.info(f"Creating memmap file ({embeddings_size_gb:.2f} GB)")
            emb = np.memmap(
                embeddings_path, dtype=np.float32, mode='w+',
                shape=(total_samples, embedding_dim)
            )
            del emb  # Close so other ranks can open
    dist.barrier()  # All ranks wait for file creation

    # All ranks open in r+ mode (concurrent writes to disjoint regions)
    embeddings = np.memmap(
        embeddings_path, dtype=np.float32, mode='r+',
        shape=(total_samples, embedding_dim)
    )

    # =============================================
    # Check for resume (per-rank checkpoints)
    # =============================================
    resume_batch_idx = 0
    resume_sample_offset = 0
    rank_checkpoint_path = output_dir / f"extraction_checkpoint_rank{global_rank}.json"

    if args.resume and rank_checkpoint_path.exists():
        with open(rank_checkpoint_path, "r") as f:
            ckpt = json.load(f)
        resume_batch_idx = ckpt["batch_idx"] + 1
        resume_sample_offset = ckpt["samples_processed"]
        logger.info(f"Rank {global_rank}: resuming from batch {resume_batch_idx}, "
                    f"sample offset {resume_sample_offset}")

    # =============================================
    # Create DataLoader for this rank's shard
    # =============================================
    dataloader = DataLoader(
        rank_dataset,
        batch_size=args.batch_size,  # Per-GPU batch size (no scaling needed)
        shuffle=False,  # Preserve order for correct memmap writes
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    total_batches = len(dataloader)
    effective_batch_size = args.batch_size * world_size
    logger.info(f"Per-GPU: {total_batches:,} batches, batch_size={args.batch_size}")
    logger.info(f"Effective total batch size: {effective_batch_size}")
    logger.info("Starting embedding extraction...")

    start_time = time.time()
    current_idx = rank_start + resume_sample_offset
    samples_processed = resume_sample_offset

    # Skip already processed batches if resuming
    dataloader_iter = iter(dataloader)
    if resume_batch_idx > 0:
        logger.info(f"Skipping {resume_batch_idx} already processed batches...")
        for _ in range(resume_batch_idx):
            next(dataloader_iter)

    with torch.no_grad():
        for batch_idx in tqdm(range(resume_batch_idx, total_batches),
                              initial=resume_batch_idx, total=total_batches,
                              desc="Extracting embeddings",
                              disable=(global_rank != 0)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break

            batch_size = batch["s2_aug1"].shape[0]

            # Move to device (non_blocking for CPU-GPU overlap)
            s2_data = batch["s2_aug1"].to(device, non_blocking=True)
            s1_data = batch["s1_aug1"].to(device, non_blocking=True)

            # Forward pass - get embeddings
            _, batch_embeddings = model(s2_data, s1_data)

            # Store embeddings at the correct global position
            end_idx = current_idx + batch_size
            embeddings[current_idx:end_idx] = batch_embeddings.cpu().numpy()

            current_idx = end_idx
            samples_processed += batch_size

            # Flush memmap periodically
            if (batch_idx + 1) % 1000 == 0:
                embeddings.flush()

            # Save per-rank checkpoint
            if (batch_idx + 1) % args.checkpoint_interval == 0:
                embeddings.flush()
                ckpt_data = {
                    "batch_idx": batch_idx,
                    "samples_processed": samples_processed,
                    "rank": global_rank,
                    "rank_start": rank_start,
                    "rank_end": rank_end,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(rank_checkpoint_path, "w") as f:
                    json.dump(ckpt_data, f)
                if global_rank == 0:
                    logger.info(f"Checkpoint saved at batch {batch_idx + 1}")

            # Log progress (rank 0 estimates total across all ranks)
            if global_rank == 0 and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rank0_speed = samples_processed / elapsed if elapsed > 0 else 0
                total_throughput = rank0_speed * world_size
                estimated_total_done = int((samples_processed / rank_total) * total_samples) if rank_total > 0 else 0
                eta_seconds = (total_samples - estimated_total_done) / total_throughput if total_throughput > 0 else 0
                logger.info(
                    f"Batch {batch_idx + 1:,}/{total_batches:,}, "
                    f"~{estimated_total_done:,}/{total_samples:,} total samples, "
                    f"Throughput: {total_throughput:.1f}/s ({rank0_speed:.1f}/s per GPU), "
                    f"ETA: {eta_seconds/3600:.1f}h"
                )

    # Final flush for this rank
    embeddings.flush()

    # =============================================
    # Synchronize all ranks before finalization
    # =============================================
    dist.barrier()

    total_time = time.time() - start_time

    # =============================================
    # Rank 0 handles finalization
    # =============================================
    if global_rank == 0:
        logger.info(f"All ranks completed in {total_time/3600:.2f} hours")
        del embeddings  # Close memmap

        # Create and save index mapping
        logger.info("Creating index mapping...")
        index_mapping_path = output_dir / "index_mapping.npy"

        index_mapping = np.zeros((total_samples, 2), dtype=np.int32)
        current_pos = 0
        for file_idx, (start, end) in enumerate(zip(dataset.file_offsets[:-1], dataset.file_offsets[1:])):
            num_rows = end - start
            if current_pos + num_rows > total_samples:
                num_rows = total_samples - current_pos
            if num_rows <= 0:
                break
            index_mapping[current_pos:current_pos + num_rows, 0] = file_idx
            index_mapping[current_pos:current_pos + num_rows, 1] = np.arange(num_rows)
            current_pos += num_rows

        logger.info(f"Saving index mapping to {index_mapping_path}")
        np.save(index_mapping_path, index_mapping)

        # Save metadata
        metadata = {
            "total_samples": total_samples,
            "embedding_dim": embedding_dim,
            "checkpoint_path": str(args.checkpoint),
            "index_dir": str(args.index_dir),
            "data_root": str(args.data_root),
            "extraction_time_seconds": total_time,
            "n_gpus": world_size,
            "batch_size_per_gpu": args.batch_size,
            "effective_batch_size": effective_batch_size,
        }
        metadata_path = output_dir / "extraction_metadata.txt"
        with open(metadata_path, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        # Clean up per-rank checkpoint files on success
        for r in range(world_size):
            ckpt_file = output_dir / f"extraction_checkpoint_rank{r}.json"
            if ckpt_file.exists():
                ckpt_file.unlink()
        logger.info("Removed checkpoint files after successful completion")

        logger.info("Done!")
        logger.info(f"  Embeddings: {embeddings_path}")
        logger.info(f"  Index mapping: {index_mapping_path}")
        logger.info(f"  Metadata: {metadata_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
