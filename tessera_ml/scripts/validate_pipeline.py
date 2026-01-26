#!/usr/bin/env python3
"""
Validate the SSL data curation pipeline on a small sample.

This script tests each stage of the pipeline:
1. Embedding extraction (100 samples)
2. Hierarchical clustering
3. Curated index sampling
4. TreeDataset compatibility

Usage:
    python scripts/validate_pipeline.py --checkpoint /path/to/checkpoint.pt

Run interactively or via:
    sbatch validate_pipeline.sbatch
"""

import argparse
import os
import sys
import logging
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add parent directory to path (ssl_data_curation is accessed as a subpackage)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate SSL data curation pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera_best_model_fsdp_20250427_084307.pt",
        help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to test with"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: temp directory)"
    )
    parser.add_argument(
        "--keep_output",
        action="store_true",
        help="Keep output directory after validation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    return parser.parse_args()


def test_imports():
    """Test that all required imports work."""
    logger.info("=" * 60)
    logger.info("STEP 0: Testing imports...")
    logger.info("=" * 60)

    errors = []

    # Test core imports
    try:
        from dataset import TreeDataset
        logger.info("  [OK] dataset.TreeDataset")
    except ImportError as e:
        errors.append(f"dataset.TreeDataset: {e}")
        logger.error(f"  [FAIL] dataset.TreeDataset: {e}")

    try:
        from models.modules import TransformerEncoder, ProjectionHead
        logger.info("  [OK] models.modules (TransformerEncoder, ProjectionHead)")
    except ImportError as e:
        errors.append(f"models.modules: {e}")
        logger.error(f"  [FAIL] models.modules: {e}")

    # Test imports needed for architecture detection
    try:
        from models.modules import (
            TemporalPositionalEncoder,
            TemporalAwarePooling,
            CustomTemporalAwarePooling
        )
        logger.info("  [OK] models.modules (TemporalAwarePooling, CustomTemporalAwarePooling)")
    except ImportError as e:
        errors.append(f"models.modules pooling classes: {e}")
        logger.error(f"  [FAIL] models.modules pooling classes: {e}")

    try:
        from models.ssl_model import MultimodalBTModel
        logger.info("  [OK] models.ssl_model")
    except ImportError as e:
        errors.append(f"models.ssl_model: {e}")
        logger.error(f"  [FAIL] models.ssl_model: {e}")

    # Test ssl_data_curation imports
    try:
        from ssl_data_curation.src import kmeans_gpu
        logger.info("  [OK] ssl_data_curation.src.kmeans_gpu")
    except ImportError as e:
        errors.append(f"kmeans_gpu: {e}")
        logger.error(f"  [FAIL] ssl_data_curation.src.kmeans_gpu: {e}")

    try:
        from ssl_data_curation.src.hierarchical_kmeans_gpu import hierarchical_kmeans
        logger.info("  [OK] ssl_data_curation.src.hierarchical_kmeans_gpu")
    except ImportError as e:
        errors.append(f"hierarchical_kmeans_gpu: {e}")
        logger.error(f"  [FAIL] ssl_data_curation.src.hierarchical_kmeans_gpu: {e}")

    try:
        from ssl_data_curation.src.clusters import HierarchicalCluster
        logger.info("  [OK] ssl_data_curation.src.clusters")
    except ImportError as e:
        errors.append(f"clusters: {e}")
        logger.error(f"  [FAIL] ssl_data_curation.src.clusters: {e}")

    if errors:
        logger.error(f"\nImport errors: {len(errors)}")
        for e in errors:
            logger.error(f"  - {e}")
        return False

    logger.info("  All imports successful!")
    return True


def infer_config_from_state_dict(state_dict):
    """Infer model configuration from checkpoint state dict."""
    config = {
        "latent_dim": 128,
        "fusion_method": "concat",
        "sample_size_s2": 40,
        "sample_size_s1": 40,
    }

    # Detect architecture first (GRU type)
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
            # in_proj_weight is (3 * d_model, d_model), nhead = d_model / head_dim
            d_model = value.shape[1]
            config["latent_dim"] = d_model // 4  # d_model = latent_dim * 4
            # Default nhead based on d_model
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

        logger.info(f"  Detected projector: {num_linear_layers} linear layers "
                    f"({config['projector_num_hidden']} hidden), "
                    f"hidden_dim={config.get('projector_hidden_dim')}, "
                    f"out_dim={config.get('projector_out_dim')}")

    return config


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


def test_checkpoint_loading(checkpoint_path, device):
    """Test that the checkpoint can be loaded with STRICT loading.

    This test verifies that:
    1. The checkpoint exists and can be read
    2. The architecture is correctly detected from checkpoint keys
    3. A model with matching architecture can be created
    4. Weights load successfully with strict=True (no missing/unexpected keys)

    If strict loading fails, it means there's an architecture mismatch,
    which would cause runtime errors during inference.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Testing checkpoint loading (STRICT MODE)...")
    logger.info("=" * 60)

    from models.modules import (
        TemporalPositionalEncoder,
        TemporalAwarePooling,
        CustomTemporalAwarePooling,
        ProjectionHead
    )
    from models.ssl_model import MultimodalBTModel
    import torch.nn as nn

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"  [FAIL] Checkpoint not found: {checkpoint_path}")
        return None, None

    logger.info(f"  Checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"  Checkpoint keys: {list(checkpoint.keys())}")

    # Get state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Clean FSDP prefixes
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
        logger.info(f"  Removed key prefix: '{key_prefix}'")

    # CRITICAL: Detect architecture from checkpoint keys
    arch_info = detect_architecture(new_state_dict)
    logger.info("  " + "-" * 40)
    logger.info("  DETECTED ARCHITECTURE:")
    logger.info(f"    Model type: {arch_info['model_type']}")
    logger.info(f"    GRU type: {arch_info['gru_type']}")
    logger.info(f"    Transformer layers: {arch_info['num_layers']}")
    logger.info(f"    Has layer_norm: {arch_info['has_layer_norm']}")
    logger.info("  " + "-" * 40)

    # Try to get config from checkpoint, otherwise infer from state dict
    config = checkpoint.get("config", {})
    if config:
        logger.info(f"  Config found with {len(config)} keys")
    else:
        logger.warning("  No config in checkpoint, inferring from state dict...")
        config = infer_config_from_state_dict(new_state_dict)
        logger.info(f"  Inferred config: {config}")

    # Add architecture info to config
    config["use_custom_gru"] = arch_info["use_custom_gru"]
    config["model_type"] = arch_info["model_type"]

    # Create Universal ProjectionHead that matches checkpoint structure
    class ProjectionHeadUniversal(nn.Module):
        """ProjectionHead that matches the checkpoint structure."""
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

    # Create Universal TransformerEncoder that supports both GRU types
    class TransformerEncoderUniversal(nn.Module):
        """TransformerEncoder that supports both GRU types."""
        def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=8,
                     dim_feedforward=4096, dropout=0.1, use_custom_gru=False):
            super().__init__()
            input_dim = band_num

            self.embedding = nn.Sequential(
                nn.Linear(input_dim, latent_dim * 4),
                nn.ReLU(),
                nn.Linear(latent_dim * 4, latent_dim * 4)
            )

            self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

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
            bands = x[:, :, :-1]
            doy = x[:, :, -1]
            bands_embedded = self.embedding(bands)
            temporal_encoding = self.temporal_encoder(doy)
            x = bands_embedded + temporal_encoding
            x = self.transformer_encoder(x)
            x = self.attn_pool(x)
            return x

    # Create model with CORRECT architecture based on detection
    latent_dim = config.get("latent_dim", 128)
    use_custom_gru = config.get("use_custom_gru", False)

    logger.info(f"  Creating model with use_custom_gru={use_custom_gru}")

    s2_enc = TransformerEncoderUniversal(
        band_num=10, latent_dim=latent_dim,
        nhead=config.get("s2_num_heads", 8),
        num_encoder_layers=config.get("s2_num_layers", 8),
        dim_feedforward=config.get("s2_dim_feedforward", 4096),
        use_custom_gru=use_custom_gru,
    ).to(device)

    s1_enc = TransformerEncoderUniversal(
        band_num=2, latent_dim=latent_dim,
        nhead=config.get("s1_num_heads", 8),
        num_encoder_layers=config.get("s1_num_layers", 8),
        dim_feedforward=config.get("s1_dim_feedforward", 4096),
        use_custom_gru=use_custom_gru,
    ).to(device)

    projector = ProjectionHeadUniversal(
        input_dim=latent_dim,
        hidden_dim=config.get("projector_hidden_dim", 16384),
        output_dim=config.get("projector_out_dim", 16384),
        num_hidden_layers=config.get("projector_num_hidden", 4),
    ).to(device)
    logger.info(f"  Creating projector: input={latent_dim}, hidden={config.get('projector_hidden_dim', 16384)}, "
                f"out={config.get('projector_out_dim', 16384)}, num_hidden={config.get('projector_num_hidden', 4)}")

    model = MultimodalBTModel(
        s2_enc, s1_enc, projector,
        fusion_method=config.get("fusion_method", "concat"),
        return_repr=True,
        latent_dim=latent_dim,
        apply_qat_representation=False,
    ).to(device)

    # CRITICAL: Load state dict with STRICT=TRUE
    # This is the key test - if this fails, there's an architecture mismatch
    try:
        model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"  [OK] Model loaded successfully with strict=True")
    except RuntimeError as e:
        logger.error(f"  [FAIL] Strict loading failed!")
        logger.error(f"  Error: {str(e)[:300]}...")

        # Provide diagnostic information
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(new_state_dict.keys())
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys

        logger.error(f"  Missing keys ({len(missing)}): {list(missing)[:5]}...")
        logger.error(f"  Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}...")
        logger.error("")
        logger.error("  This means the model architecture doesn't match the checkpoint!")
        logger.error(f"  Detected: {arch_info['model_type']} with {arch_info['gru_type']}")
        logger.error("  Ensure TransformerEncoderUniversal is using the correct GRU type.")
        return None, None

    model.eval()
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def test_dataset_loading(n_samples):
    """Test that the dataset can be loaded."""
    logger.info("=" * 60)
    logger.info("STEP 2: Testing dataset loading...")
    logger.info("=" * 60)

    from dataset import TreeDataset

    index_dir = "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid"
    data_root = "/scratch/users/psinghal/time_series"

    logger.info(f"  Index dir: {index_dir}")
    logger.info(f"  Data root: {data_root}")

    dataset = TreeDataset(
        index_dir=index_dir,
        data_dir=data_root,
        year=None,
        years=list(range(2016, 2025)),
        cache_size=10,
        sample_size_s2=40,
        sample_size_s1=40,
        normalize=True,
    )

    logger.info(f"  [OK] Dataset created with {len(dataset)} samples")

    # Test loading a sample
    logger.info("  Testing sample loading...")
    sample = dataset[0]
    logger.info(f"  [OK] Sample 0 loaded")
    logger.info(f"    s2_aug1 shape: {sample['s2_aug1'].shape}")
    logger.info(f"    s1_aug1 shape: {sample['s1_aug1'].shape}")

    return dataset


def test_embedding_extraction(model, dataset, n_samples, device, output_dir):
    """Test embedding extraction on a small subset."""
    logger.info("=" * 60)
    logger.info(f"STEP 3: Testing embedding extraction ({n_samples} samples)...")
    logger.info("=" * 60)

    # Create subset
    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=min(32, n_samples),
        shuffle=False,
        num_workers=0,  # Use 0 for validation to avoid issues
        pin_memory=True,
    )

    embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            s2_data = batch["s2_aug1"].to(device)
            s1_data = batch["s1_aug1"].to(device)

            _, batch_embeddings = model(s2_data, s1_data)
            embeddings.append(batch_embeddings.cpu().numpy())

            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}: {batch_embeddings.shape}")

    embeddings = np.vstack(embeddings)
    logger.info(f"  [OK] Extracted embeddings: {embeddings.shape}")

    # Check for NaN/Inf
    if np.isnan(embeddings).any():
        logger.error("  [FAIL] Embeddings contain NaN values!")
        return None
    if np.isinf(embeddings).any():
        logger.error("  [FAIL] Embeddings contain Inf values!")
        return None

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"  Saved to: {embeddings_path}")

    # Create fake index mapping for validation
    index_mapping = np.array([[0, i] for i in range(n_samples)], dtype=np.int32)
    index_mapping_path = output_dir / "index_mapping.npy"
    np.save(index_mapping_path, index_mapping)

    return embeddings


def test_clustering(embeddings, output_dir, device):
    """Test hierarchical clustering."""
    logger.info("=" * 60)
    logger.info("STEP 4: Testing hierarchical clustering...")
    logger.info("=" * 60)

    from ssl_data_curation.src.hierarchical_kmeans_gpu import hierarchical_kmeans

    n_samples = embeddings.shape[0]

    # Use small cluster counts for validation (4 levels)
    n_clusters = [min(10, n_samples // 2), min(5, n_samples // 4), min(3, n_samples // 8), min(2, n_samples // 16)]
    n_levels = 4

    logger.info(f"  n_samples: {n_samples}")
    logger.info(f"  n_levels: {n_levels}")
    logger.info(f"  n_clusters: {n_clusters}")

    # Convert to tensor
    data = torch.tensor(embeddings, dtype=torch.float32, device=device)

    # Run clustering
    cluster_results = hierarchical_kmeans(
        data=data,
        n_clusters=n_clusters,
        n_levels=n_levels,
        verbose=True,
    )

    logger.info(f"  [OK] Clustering completed")

    # Save results
    for level_idx, level_result in enumerate(cluster_results):
        level_num = level_idx + 1
        level_dir = output_dir / f"level{level_num}"
        level_dir.mkdir(parents=True, exist_ok=True)

        clusters = level_result["clusters"]
        centroids = level_result["centroids"]

        # Sort clusters by distance to centroid
        if level_idx == 0:
            sorted_clusters = []
            for i, cluster_indices in enumerate(clusters):
                if len(cluster_indices) > 0:
                    cluster_data = data[cluster_indices]
                    centroid = centroids[i:i+1]
                    distances = torch.cdist(cluster_data, centroid).flatten()
                    sorted_order = torch.argsort(distances).cpu().numpy()
                    sorted_cluster = cluster_indices[sorted_order]
                    sorted_clusters.append(sorted_cluster)
                else:
                    sorted_clusters.append(np.array([], dtype=np.int64))
            sorted_clusters = np.array(sorted_clusters, dtype=object)
        else:
            sorted_clusters = np.array(clusters, dtype=object)

        np.save(level_dir / "sorted_clusters.npy", sorted_clusters)
        np.save(level_dir / "clusters.npy", np.array(clusters, dtype=object))
        np.save(level_dir / "centroids.npy", centroids.cpu().numpy())
        np.save(level_dir / "assignment.npy", level_result["assignment"])

        logger.info(f"  Level {level_num}: {len(clusters)} clusters")

    return cluster_results


def test_sampling(output_dir, n_samples, target_size):
    """Test sampling from clusters."""
    logger.info("=" * 60)
    logger.info(f"STEP 5: Testing sampling (target_size={target_size})...")
    logger.info("=" * 60)

    # Load clusters
    level1_clusters = np.load(output_dir / "level1" / "sorted_clusters.npy", allow_pickle=True)

    logger.info(f"  Loaded {len(level1_clusters)} clusters")

    # Simple sampling: take proportionally from each cluster
    sampled_indices = []
    cluster_sizes = np.array([len(c) for c in level1_clusters])
    total_available = cluster_sizes.sum()

    for i, cluster in enumerate(level1_clusters):
        if len(cluster) == 0:
            continue
        n_from_cluster = max(1, int(target_size * len(cluster) / total_available))
        n_from_cluster = min(n_from_cluster, len(cluster))
        sampled = cluster[:n_from_cluster]
        sampled_indices.extend(sampled)

    sampled_indices = np.array(sampled_indices[:target_size], dtype=np.int64)

    logger.info(f"  [OK] Sampled {len(sampled_indices)} indices")

    # Validate indices
    if len(sampled_indices) == 0:
        logger.error("  [FAIL] No indices sampled!")
        return None

    max_idx = n_samples - 1
    invalid = sampled_indices > max_idx
    if invalid.any():
        logger.error(f"  [FAIL] {invalid.sum()} invalid indices found")
        return None

    # Save sampled indices
    np.save(output_dir / "sampled_indices.npy", sampled_indices)

    return sampled_indices


def test_curated_index_creation(sampled_indices, output_dir):
    """Test creating a curated parquet index."""
    logger.info("=" * 60)
    logger.info("STEP 6: Testing curated index creation...")
    logger.info("=" * 60)

    import pandas as pd

    original_index_dir = Path("/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid")
    curated_dir = output_dir / "curated_index_test"
    curated_dir.mkdir(parents=True, exist_ok=True)

    # Load first parquet file
    parquet_files = sorted(list(original_index_dir.glob("part.*.parquet")))
    if not parquet_files:
        logger.error(f"  [FAIL] No parquet files found in {original_index_dir}")
        return False

    logger.info(f"  Loading from: {parquet_files[0]}")
    df = pd.read_parquet(parquet_files[0])
    logger.info(f"  Original schema: {list(df.columns)}")

    # Get rows for sampled indices (just use first few rows for test)
    valid_indices = sampled_indices[sampled_indices < len(df)]
    if len(valid_indices) == 0:
        logger.warning("  No valid indices in first parquet file, using first 10 rows")
        curated_df = df.head(10)
    else:
        curated_df = df.iloc[valid_indices[:min(len(valid_indices), 100)]]

    # Save curated index
    output_path = curated_dir / "part.0000.parquet"
    curated_df.to_parquet(output_path, index=False)
    logger.info(f"  [OK] Created curated index: {output_path}")
    logger.info(f"  Curated samples: {len(curated_df)}")

    return True


def test_curated_dataset_loading(output_dir):
    """Test that the curated index works with TreeDataset."""
    logger.info("=" * 60)
    logger.info("STEP 7: Testing curated dataset loading...")
    logger.info("=" * 60)

    from dataset import TreeDataset

    curated_dir = output_dir / "curated_index_test"
    data_root = "/scratch/users/psinghal/time_series"

    try:
        dataset = TreeDataset(
            index_dir=str(curated_dir),
            data_dir=data_root,
            year=None,
            years=list(range(2016, 2025)),
            cache_size=5,
            sample_size_s2=40,
            sample_size_s1=40,
            normalize=True,
        )

        logger.info(f"  [OK] Curated dataset created with {len(dataset)} samples")

        # Try loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"  [OK] Sample 0 loaded successfully")
            logger.info(f"    s2_aug1: {sample['s2_aug1'].shape}")
            logger.info(f"    s1_aug1: {sample['s1_aug1'].shape}")

        return True
    except Exception as e:
        logger.error(f"  [FAIL] Failed to load curated dataset: {e}")
        return False


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("SSL Data Curation Pipeline Validation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"N samples: {args.n_samples}")
    logger.info(f"Device: {args.device}")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="ssl_curation_validate_"))
        cleanup = not args.keep_output

    logger.info(f"Output dir: {output_dir}")

    results = {}

    try:
        # Step 0: Test imports
        results["imports"] = test_imports()
        if not results["imports"]:
            logger.error("\nValidation FAILED at imports step")
            return 1

        # Step 1: Test checkpoint loading
        model, config = test_checkpoint_loading(args.checkpoint, args.device)
        results["checkpoint"] = model is not None
        if not results["checkpoint"]:
            logger.error("\nValidation FAILED at checkpoint loading step")
            return 1

        # Step 2: Test dataset loading
        dataset = test_dataset_loading(args.n_samples)
        results["dataset"] = dataset is not None
        if not results["dataset"]:
            logger.error("\nValidation FAILED at dataset loading step")
            return 1

        # Step 3: Test embedding extraction
        embeddings = test_embedding_extraction(
            model, dataset, args.n_samples, args.device, output_dir
        )
        results["embeddings"] = embeddings is not None
        if not results["embeddings"]:
            logger.error("\nValidation FAILED at embedding extraction step")
            return 1

        # Step 4: Test clustering
        cluster_results = test_clustering(embeddings, output_dir, args.device)
        results["clustering"] = cluster_results is not None
        if not results["clustering"]:
            logger.error("\nValidation FAILED at clustering step")
            return 1

        # Step 5: Test sampling
        target_size = min(args.n_samples // 2, 50)
        sampled_indices = test_sampling(output_dir, args.n_samples, target_size)
        results["sampling"] = sampled_indices is not None
        if not results["sampling"]:
            logger.error("\nValidation FAILED at sampling step")
            return 1

        # Step 6: Test curated index creation
        results["index_creation"] = test_curated_index_creation(sampled_indices, output_dir)
        if not results["index_creation"]:
            logger.error("\nValidation FAILED at index creation step")
            return 1

        # Step 7: Test curated dataset loading
        results["curated_loading"] = test_curated_dataset_loading(output_dir)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        all_passed = True
        for step, passed in results.items():
            status = "[OK]" if passed else "[FAIL]"
            logger.info(f"  {status} {step}")
            if not passed:
                all_passed = False

        if all_passed:
            logger.info("\n" + "=" * 60)
            logger.info("ALL VALIDATION STEPS PASSED!")
            logger.info("=" * 60)
            logger.info("\nYou can now run the full pipeline:")
            logger.info("  sbatch extract_embeddings.sbatch")
            logger.info("  sbatch --dependency=afterok:<JOB_ID> run_clustering.sbatch")
            logger.info("  sbatch --dependency=afterok:<JOB_ID> sample_curated_index.sbatch <target_size>")
            return 0
        else:
            logger.error("\nSome validation steps FAILED")
            return 1

    finally:
        if cleanup:
            logger.info(f"\nCleaning up: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            logger.info(f"\nOutput kept at: {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
