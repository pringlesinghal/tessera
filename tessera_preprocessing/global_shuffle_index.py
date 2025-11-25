#!/usr/bin/env python3
"""
global_shuffle_index.py

Performs a true global shuffle of the tree index dataset.
It reads the partially-sorted shards, sorts them globally by 'rand_sort',
and writes out a new, perfectly randomized set of Parquet files.

Requires: dask, pandas, pyarrow
"""

import argparse
from pathlib import Path
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Globally sort/shuffle the tree index."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing original part-*.parquet files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the new global shuffled index",
    )
    parser.add_argument(
        "--temp_dir",
        default="/scratch/users/psinghal/dask_temp",
        help="Scratch space for Dask shuffling",
    )
    parser.add_argument(
        "--memory_limit", default="32GB", help="Memory limit for Dask workers"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        print(
            f"[WARN] Output directory {output_dir} exists. Please remove or specify a new one."
        )
        return

    # 1. Setup Dask Client (Local Cluster)
    # This manages memory and spilling to disk if sorting exceeds RAM
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit=args.memory_limit,
        local_directory=args.temp_dir,
    )
    client = Client(cluster)
    print(f"Dask Dashboard: {client.dashboard_link}")

    try:
        # 2. Lazy Load Data
        print(f"Reading parquet from {input_dir}...")
        df = dd.read_parquet(input_dir / "*.parquet", engine="pyarrow")

        # 3. Global Sort
        # Setting the index to 'rand_sort' forces a global shuffle/sort.
        # This is the heavy lifting step.
        print("Sorting globally by 'rand_sort'...")
        df_sorted = df.set_index("rand_sort", sorted=False)

        # 4. Repartition (Optional optimization)
        # Ensure partitions are ~100MB each for optimal reading later
        # df_sorted = df_sorted.repartition(partition_size="100MB")

        # 5. Write Output
        print(f"Writing sorted index to {output_dir}...")
        # reset_index(drop=True) because we don't need 'rand_sort' as the index anymore, just as a column (or dropped)
        # Actually, keeping it as index or column is fine, but usually we just want a flat table.
        # We drop=False to keep the rand_sort value, or True if we don't care anymore.
        # Let's keep it.
        df_final = df_sorted.reset_index()

        df_final.to_parquet(
            output_dir, engine="pyarrow", compression="snappy", write_index=False
        )

        print("Global shuffle completed successfully.")

    except Exception as e:
        print(f"[ERROR] Sorting failed: {e}")
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
