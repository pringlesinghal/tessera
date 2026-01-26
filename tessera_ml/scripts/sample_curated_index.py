#!/usr/bin/env python3
"""
Sample from hierarchical clusters to create a curated dataset index.

This script:
1. Loads hierarchical cluster results
2. Performs hierarchical sampling to select target_size samples
3. Maps sampled indices back to original parquet index format
4. Creates new parquet index files compatible with TreeDataset

Usage:
    python sample_curated_index.py \
        --cluster_dir /path/to/curation_outputs \
        --index_mapping_path /path/to/index_mapping.npy \
        --original_index_dir /path/to/original_parquet_index \
        --output_dir /path/to/curated_index \
        --target_size 100000 \
        --strategy r
"""

import argparse
import os
import sys
import logging
import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add parent directory to path (ssl_data_curation is accessed as a subpackage)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample curated dataset from hierarchical clusters")
    parser.add_argument(
        "--cluster_dir",
        type=str,
        default="curation_outputs",
        help="Directory containing clustering results (with level1/, level2/, etc.)"
    )
    parser.add_argument(
        "--index_mapping_path",
        type=str,
        default="curation_outputs/index_mapping.npy",
        help="Path to index_mapping.npy file"
    )
    parser.add_argument(
        "--original_index_dir",
        type=str,
        default="/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid",
        help="Path to original parquet index directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save curated index (e.g., curated_index_100000)"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        required=True,
        help="Number of samples to select for the curated dataset"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="r",
        choices=["r", "c"],
        help="Sampling strategy: 'r' for random, 'c' for closest-to-centroid"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Maximum number of times a sample can be selected (default: 1)"
    )
    parser.add_argument(
        "--rows_per_file",
        type=int,
        default=50000,
        help="Number of rows per output parquet file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_hierarchical_clusters(cluster_dir):
    """Load hierarchical clusters from directory structure."""
    try:
        from ssl_data_curation.src.clusters import HierarchicalCluster
        logger.info("Loading hierarchical clusters using HierarchicalCluster class...")
        clusters = HierarchicalCluster.from_file(
            cluster_path=cluster_dir,
            cluster_fname="sorted_clusters.npy"
        )
        return clusters
    except Exception as e:
        logger.warning(f"Failed to load using HierarchicalCluster: {e}")
        logger.info("Falling back to manual loading...")
        return load_clusters_manual(cluster_dir)


def load_clusters_manual(cluster_dir):
    """Manually load cluster results from directory."""
    cluster_dir = Path(cluster_dir)

    # Find all level directories
    level_dirs = sorted([d for d in cluster_dir.iterdir() if d.is_dir() and d.name.startswith("level")])
    n_levels = len(level_dirs)

    if n_levels == 0:
        raise FileNotFoundError(f"No level directories found in {cluster_dir}")

    logger.info(f"Found {n_levels} levels")

    # Load cluster results
    cluster_results = []
    for level_dir in level_dirs:
        clusters_path = level_dir / "sorted_clusters.npy"
        if not clusters_path.exists():
            clusters_path = level_dir / "clusters.npy"

        clusters = np.load(clusters_path, allow_pickle=True)
        cluster_results.append({"clusters": clusters})
        logger.info(f"  {level_dir.name}: {len(clusters)} clusters")

    return cluster_results


def hierarchical_sample_simple(cluster_results, target_size, strategy="r", multiplier=1, seed=42):
    """
    Simple hierarchical sampling implementation.

    This performs balanced sampling across clusters at the deepest level.
    """
    np.random.seed(seed)

    n_levels = len(cluster_results)
    logger.info(f"Sampling {target_size} points from {n_levels}-level hierarchy")

    # Get the leaf-level clusters (level 1 in the hierarchy)
    leaf_clusters = cluster_results[0]["clusters"]
    n_clusters = len(leaf_clusters)

    # Calculate how many samples to take from each cluster
    cluster_sizes = np.array([len(c) for c in leaf_clusters])
    total_available = cluster_sizes.sum() * multiplier

    if target_size > total_available:
        logger.warning(f"Target size {target_size} exceeds available samples {total_available}")
        target_size = total_available

    # Distribute samples across clusters proportionally, with minimum 1 per non-empty cluster
    non_empty_clusters = np.where(cluster_sizes > 0)[0]
    n_non_empty = len(non_empty_clusters)

    if n_non_empty == 0:
        raise ValueError("All clusters are empty!")

    # Start with equal distribution, respecting cluster sizes
    samples_per_cluster = np.zeros(n_clusters, dtype=int)

    # Calculate proportional allocation
    for i in non_empty_clusters:
        max_from_cluster = cluster_sizes[i] * multiplier
        proportional = int(target_size * (cluster_sizes[i] / cluster_sizes.sum()))
        samples_per_cluster[i] = min(proportional, max_from_cluster)

    # Distribute remainder
    current_total = samples_per_cluster.sum()
    remainder = target_size - current_total

    if remainder > 0:
        # Add to clusters that can still accommodate more
        for _ in range(remainder):
            # Find clusters with room
            can_add = []
            for i in non_empty_clusters:
                max_from_cluster = cluster_sizes[i] * multiplier
                if samples_per_cluster[i] < max_from_cluster:
                    can_add.append(i)

            if not can_add:
                break

            # Add to a random cluster that has room
            chosen = np.random.choice(can_add)
            samples_per_cluster[chosen] += 1

    logger.info(f"Sampling distribution: min={samples_per_cluster.min()}, max={samples_per_cluster.max()}, "
                f"mean={samples_per_cluster.mean():.1f}")

    # Sample from each cluster
    sampled_indices = []
    for cluster_idx in range(n_clusters):
        n_samples = samples_per_cluster[cluster_idx]
        if n_samples == 0:
            continue

        cluster_points = leaf_clusters[cluster_idx]
        if len(cluster_points) == 0:
            continue

        if strategy == "r":
            # Random sampling
            if n_samples <= len(cluster_points) * multiplier:
                if multiplier == 1:
                    selected = np.random.choice(cluster_points, n_samples, replace=False)
                else:
                    # With multiplier > 1, we can select the same point multiple times
                    n_unique = min(n_samples, len(cluster_points))
                    selected = np.random.choice(cluster_points, n_unique, replace=False)
                    if n_samples > n_unique:
                        extra = np.random.choice(cluster_points, n_samples - n_unique, replace=True)
                        selected = np.concatenate([selected, extra])
            else:
                selected = np.tile(cluster_points, multiplier)[:n_samples]
        else:
            # Closest to centroid (clusters should be pre-sorted)
            if n_samples <= len(cluster_points):
                selected = cluster_points[:n_samples]
            else:
                # Need more samples than available - replicate
                selected = np.tile(cluster_points, (n_samples // len(cluster_points)) + 1)[:n_samples]

        sampled_indices.append(selected)

    sampled_indices = np.concatenate(sampled_indices).astype(np.int64)

    # Shuffle final selection
    np.random.shuffle(sampled_indices)

    logger.info(f"Sampled {len(sampled_indices)} points (target: {target_size})")

    return sampled_indices


def create_curated_parquet_index(sampled_indices, index_mapping, original_index_dir, output_dir, rows_per_file):
    """Create new parquet index files from sampled indices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_index_dir = Path(original_index_dir)

    # Group sampled indices by their original parquet file
    # index_mapping[i] = [file_idx, local_row_idx]
    file_to_rows = defaultdict(list)
    for global_idx in sampled_indices:
        file_idx, local_row = index_mapping[global_idx]
        file_to_rows[file_idx].append(local_row)

    # Load original parquet files and collect sampled rows
    original_files = sorted(list(original_index_dir.glob("part.*.parquet")))
    if not original_files:
        raise FileNotFoundError(f"No parquet files found in {original_index_dir}")

    logger.info(f"Loading samples from {len(file_to_rows)} original parquet files...")

    all_sampled_rows = []
    for file_idx in sorted(file_to_rows.keys()):
        row_indices = file_to_rows[file_idx]
        df = pd.read_parquet(original_files[file_idx])

        # Select the sampled rows
        sampled_df = df.iloc[row_indices].copy()
        all_sampled_rows.append(sampled_df)

        if len(all_sampled_rows) % 10 == 0:
            logger.info(f"  Processed {len(all_sampled_rows)} files...")

    # Combine all sampled rows
    logger.info("Combining sampled rows...")
    curated_df = pd.concat(all_sampled_rows, ignore_index=True)

    # Shuffle the combined dataframe
    logger.info("Shuffling curated dataset...")
    curated_df = curated_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Verify schema
    required_columns = ["tile_id", "row", "col"]
    for col in required_columns:
        if col not in curated_df.columns:
            raise ValueError(f"Missing required column: {col}")

    logger.info(f"Curated dataset: {len(curated_df)} samples")
    logger.info(f"Columns: {list(curated_df.columns)}")

    # Write as multiple parquet files
    n_files = (len(curated_df) + rows_per_file - 1) // rows_per_file
    logger.info(f"Writing {n_files} parquet files to {output_dir}...")

    for i in range(n_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, len(curated_df))
        chunk = curated_df.iloc[start_idx:end_idx]

        output_path = output_dir / f"part.{i:04d}.parquet"
        chunk.to_parquet(output_path, index=False)

    logger.info(f"Created {n_files} parquet files with {len(curated_df)} total samples")

    return len(curated_df)


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)

    logger.info(f"Configuration:")
    logger.info(f"  Cluster directory: {args.cluster_dir}")
    logger.info(f"  Index mapping: {args.index_mapping_path}")
    logger.info(f"  Original index: {args.original_index_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Target size: {args.target_size}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Multiplier: {args.multiplier}")

    # Load index mapping
    logger.info(f"Loading index mapping from {args.index_mapping_path}")
    index_mapping = np.load(args.index_mapping_path)
    logger.info(f"Index mapping shape: {index_mapping.shape}")

    # Load hierarchical clusters
    logger.info(f"Loading clusters from {args.cluster_dir}")
    try:
        # Try to use the HierarchicalCluster class
        from ssl_data_curation.src.clusters import HierarchicalCluster
        from ssl_data_curation.src.hierarchical_sampling import hierarchical_sampling

        clusters = HierarchicalCluster.from_file(
            cluster_path=args.cluster_dir,
            cluster_fname="sorted_clusters.npy"
        )

        logger.info("Using library hierarchical_sampling function...")
        sampled_indices = hierarchical_sampling(
            clusters=clusters,
            target_size=args.target_size,
            multiplier=args.multiplier,
            sampling_strategy=args.strategy,
        )
    except Exception as e:
        logger.warning(f"Failed to use library sampling: {e}")
        logger.info("Falling back to simple sampling implementation...")

        # Load cluster results manually
        cluster_results = load_clusters_manual(args.cluster_dir)

        # Use simple sampling
        sampled_indices = hierarchical_sample_simple(
            cluster_results=cluster_results,
            target_size=args.target_size,
            strategy=args.strategy,
            multiplier=args.multiplier,
            seed=args.seed,
        )

    logger.info(f"Sampled {len(sampled_indices)} indices")

    # Validate sampled indices
    max_idx = index_mapping.shape[0] - 1
    invalid_mask = sampled_indices > max_idx
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        logger.warning(f"Found {n_invalid} invalid indices (> {max_idx}), filtering them out")
        sampled_indices = sampled_indices[~invalid_mask]

    # Create curated parquet index
    logger.info("Creating curated parquet index...")
    start_time = time.time()

    n_samples = create_curated_parquet_index(
        sampled_indices=sampled_indices,
        index_mapping=index_mapping,
        original_index_dir=args.original_index_dir,
        output_dir=output_dir,
        rows_per_file=args.rows_per_file,
    )

    creation_time = time.time() - start_time
    logger.info(f"Index creation completed in {creation_time:.2f}s")

    # Save metadata
    metadata = {
        "target_size": args.target_size,
        "actual_size": n_samples,
        "strategy": args.strategy,
        "multiplier": args.multiplier,
        "seed": args.seed,
        "cluster_dir": str(args.cluster_dir),
        "original_index_dir": str(args.original_index_dir),
        "creation_time_seconds": creation_time,
    }
    metadata_path = output_dir / "curation_metadata.txt"
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    # Also save the sampled indices for reference
    np.save(output_dir / "sampled_indices.npy", sampled_indices)

    logger.info("Done!")
    logger.info(f"  Curated index: {output_dir}")
    logger.info(f"  Samples: {n_samples}")
    logger.info(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
