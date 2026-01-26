#!/usr/bin/env python3
"""
Run hierarchical k-means clustering on extracted embeddings.

This script:
1. Loads embeddings from embeddings.npy
2. Runs hierarchical k-means clustering
3. Saves cluster assignments and hierarchy for later sampling

Usage:
    python run_clustering.py \
        --embeddings_path /path/to/embeddings.npy \
        --output_dir /path/to/output \
        --n_levels 2 \
        --n_clusters 1000 100
"""

import argparse
import os
import sys
import logging
import time
import pickle
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path (ssl_data_curation is accessed as a subpackage)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run hierarchical k-means clustering")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="curation_outputs/embeddings.npy",
        help="Path to embeddings.npy file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="curation_outputs",
        help="Directory to save clustering results"
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=4,
        help="Number of hierarchical levels"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        nargs="+",
        default=[100000, 5000, 500, 100],
        help="Number of clusters at each level"
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[1, 10, 5, 3],
        help="Points sampled PER CLUSTER during resampling per level"
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=10,
        help="Number of resampling iterations"
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="closest",
        choices=["closest", "random"],
        help="Strategy for sampling in resampling steps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for clustering"
    )
    parser.add_argument(
        "--use_resampling",
        action="store_true",
        help="Use k-means with resampling (slower but potentially better)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate arguments
    if len(args.n_clusters) != args.n_levels:
        raise ValueError(f"n_clusters ({len(args.n_clusters)}) must match n_levels ({args.n_levels})")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load embeddings (memory-mapped for large files)
    logger.info(f"Loading embeddings from {args.embeddings_path}")
    embeddings_size = os.path.getsize(args.embeddings_path) / (1024**3)
    logger.info(f"Embeddings file size: {embeddings_size:.2f} GB")

    if embeddings_size > 50:  # Use mmap for files > 50GB
        logger.info("Using memory-mapped loading for large embeddings file")
        embeddings = np.load(args.embeddings_path, mmap_mode='r')
    else:
        embeddings = np.load(args.embeddings_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    n_samples, embedding_dim = embeddings.shape

    # sample_sizes is per-cluster, not total
    # Rule of thumb: ~half the average cluster size at each level
    if len(args.sample_sizes) != args.n_levels:
        raise ValueError(f"sample_sizes ({len(args.sample_sizes)}) must match n_levels ({args.n_levels})")

    # Log expected cluster sizes for reference
    avg_cluster_size_l1 = n_samples / args.n_clusters[0]
    logger.info(f"  Level 1: {args.n_clusters[0]} clusters, ~{avg_cluster_size_l1:.1f} points/cluster, sampling {args.sample_sizes[0]}/cluster")

    logger.info(f"Clustering configuration:")
    logger.info(f"  n_levels: {args.n_levels}")
    logger.info(f"  n_clusters: {args.n_clusters}")
    logger.info(f"  sample_sizes: {args.sample_sizes}")
    logger.info(f"  n_resamples: {args.n_resamples}")
    logger.info(f"  sample_strategy: {args.sample_strategy}")
    logger.info(f"  use_resampling: {args.use_resampling}")

    # Convert to tensor and move to device
    logger.info("Converting embeddings to tensor...")
    data = torch.tensor(embeddings, dtype=torch.float32, device=device)

    # Import clustering functions
    try:
        from ssl_data_curation.src.hierarchical_kmeans_gpu import hierarchical_kmeans, hierarchical_kmeans_with_resampling
        logger.info("Using GPU-accelerated hierarchical k-means")
    except ImportError as e:
        logger.error(f"Failed to import hierarchical k-means: {e}")
        logger.error("Make sure the ssl_data_curation library is installed correctly")
        raise

    # Run clustering
    start_time = time.time()

    if args.use_resampling:
        logger.info("Running hierarchical k-means with resampling...")
        cluster_results = hierarchical_kmeans_with_resampling(
            data=data,
            n_clusters=args.n_clusters,
            n_levels=args.n_levels,
            sample_sizes=args.sample_sizes,
            n_resamples=args.n_resamples,
            sample_strategy=args.sample_strategy,
            verbose=True,
        )
    else:
        logger.info("Running hierarchical k-means (no resampling)...")
        cluster_results = hierarchical_kmeans(
            data=data,
            n_clusters=args.n_clusters,
            n_levels=args.n_levels,
            verbose=True,
        )

    clustering_time = time.time() - start_time
    logger.info(f"Clustering completed in {clustering_time:.2f}s")

    # Save results
    logger.info("Saving clustering results...")

    # Save the raw cluster results as pickle (for compatibility with HierarchicalCluster.from_dict)
    results_path = output_dir / "cluster_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(cluster_results, f)
    logger.info(f"Saved cluster results to {results_path}")

    # Also save in the format expected by the hierarchical_sampling script
    # Create directory structure: level1/sorted_clusters.npy, level2/sorted_clusters.npy, etc.
    for level_idx, level_result in enumerate(cluster_results):
        level_num = level_idx + 1
        level_dir = output_dir / f"level{level_num}"
        level_dir.mkdir(parents=True, exist_ok=True)

        # Save clusters (indices of points in each cluster)
        clusters = level_result["clusters"]

        # Sort clusters by distance to centroid for closest-to-centroid sampling
        centroids = level_result["centroids"]
        if level_idx == 0:
            # For first level, sort points within each cluster by distance to centroid
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
            # For higher levels, clusters contain indices into previous level's clusters
            sorted_clusters = np.array(clusters, dtype=object)

        np.save(level_dir / "sorted_clusters.npy", sorted_clusters)
        np.save(level_dir / "clusters.npy", np.array(clusters, dtype=object))
        np.save(level_dir / "centroids.npy", centroids.cpu().numpy() if torch.is_tensor(centroids) else centroids)
        np.save(level_dir / "assignment.npy", level_result["assignment"])

        logger.info(f"  Level {level_num}: {len(clusters)} clusters saved to {level_dir}")

    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "embedding_dim": embedding_dim,
        "n_levels": args.n_levels,
        "n_clusters": args.n_clusters,
        "sample_sizes": args.sample_sizes,
        "n_resamples": args.n_resamples,
        "sample_strategy": args.sample_strategy,
        "use_resampling": args.use_resampling,
        "clustering_time_seconds": clustering_time,
    }
    metadata_path = output_dir / "clustering_metadata.txt"
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    logger.info("Done!")
    logger.info(f"  Cluster results: {results_path}")
    logger.info(f"  Level directories: level1/, level2/, ...")
    logger.info(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
