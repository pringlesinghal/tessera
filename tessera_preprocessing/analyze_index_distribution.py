#!/usr/bin/env python3
"""
analyze_index_distribution.py

Scans the shuffled tree index Parquet dataset and reports how many tree pixels exist
per 20 km tile. Outputs summary statistics and a histogram (textual + optional plot).
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc


def build_tile_counts(index_dir: Path, batch_size: int = 2_000_000) -> dict:
    dataset = ds.dataset(index_dir, format="parquet")
    scanner = dataset.scanner(columns=["tile_id"], batch_size=batch_size)

    counts = defaultdict(int)
    total_batches = 0

    for batch in scanner.to_batches():
        total_batches += 1
        tile_ids = batch.column(0)
        vc = pc.value_counts(tile_ids)
        values = vc.field("values")
        freqs = vc.field("counts")
        for tile, freq in zip(values.to_pylist(), freqs.to_pylist()):
            counts[tile] += int(freq)

    print(f"Processed {total_batches} batches across {len(counts)} tiles.")
    return counts


def print_summary(stats: np.ndarray) -> None:
    print("\n--- Tree Count Summary ---")
    print(f"Tiles analyzed        : {len(stats)}")
    print(f"Min trees per tile    : {stats.min():,}")
    print(f"Median trees per tile : {np.median(stats):,.0f}")
    print(f"Mean trees per tile   : {stats.mean():,.0f}")
    print(f"95th percentile       : {np.percentile(stats, 95):,.0f}")
    print(f"Max trees per tile    : {stats.max():,}")


def print_histogram(values: np.ndarray, bins: int) -> None:
    hist, bin_edges = np.histogram(values, bins=bins)
    print("\n--- Histogram (textual) ---")
    for idx in range(len(hist)):
        left = int(bin_edges[idx])
        right = int(bin_edges[idx + 1])
        count = hist[idx]
        print(f"{left:>10,} - {right:>10,}: {count:>8}")


def save_plot(values: np.ndarray, bins: int, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins)
    plt.xlabel("Tree pixels per tile")
    plt.ylabel("Number of tiles")
    plt.title("Distribution of tree counts per tile")
    plt.tight_layout()
    plt.savefig(path)
    print(f"Histogram saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tree count distribution across tiles."
    )
    parser.add_argument("--index_dir", required=True, help="Directory of shuffled Parquet index.")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins.")
    parser.add_argument("--batch_size", type=int, default=2_000_000, help="Rows per scan batch.")
    parser.add_argument("--plot", type=str, default=None, help="Optional path to save histogram image.")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    tile_counts = build_tile_counts(index_dir, args.batch_size)
    values = np.array(list(tile_counts.values()), dtype=np.int64)
    print_summary(values)
    print_histogram(values, args.bins)

    if args.plot:
        save_plot(values, args.bins, Path(args.plot))


if __name__ == "__main__":
    main()


