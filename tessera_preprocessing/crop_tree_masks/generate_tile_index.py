#!/usr/bin/env python3
"""
generate_tile_index.py

This script processes 100km MGRS tree masks to create a global, shuffled index of tree pixels.
It performs three main steps:
1.  **Subtiling**: Splits each 100km x 100km mask (10,000 px) into 25 subtiles (2,000 px each),
    aligning with the downloader's 20km grid.
2.  **Indexing**: Extracts (row, col) coordinates for every tree pixel relative to the 20km subtile.
3.  **Sharding**: Writes the indices to Parquet files with a random sort key to enable
    globally random sampling during training.

Output Structure:
    output_dir/
    ├── part-0000.parquet  (Contains ~N tree pixels from various tiles)
    ├── part-0001.parquet
    ...
    └── _metadata          (Parquet dataset metadata)

Schema:
    - tile_id: string (e.g., "zone32643_r16_c4_sr0_sc0")
    - row: uint16 (0-1999)
    - col: uint16 (0-1999)
    - rand_sort: float64 (for random access)
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import numpy as np
import rasterio
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import re

# --- Configuration ---
SUBTILE_SIZE = 2000  # 20km @ 10m resolution
PARENT_SIZE = 10000  # 100km @ 10m resolution
GRID_N = 5           # 5x5 grid of subtiles

def parse_tile_id(filename):
    """Extracts base tile ID from filename (e.g., 'zone32643_r16_c4')."""
    match = re.search(r"(zone\d+_r\d+_c\d+)", filename)
    return match.group(1) if match else None

def process_single_mask(mask_path, valid_subtiles=None):
    """
    Reads a 100km mask, splits it into 25 subtiles, and returns a DataFrame
    of tree pixels with their subtile IDs and local coordinates.
    
    Args:
        mask_path (Path): Path to 100km TIF mask.
        valid_subtiles (set, optional): Set of tile_ids (strings) to keep. 
                                        If None, keep all.
    """
    base_name = parse_tile_id(mask_path.name)
    if not base_name:
        print(f"[WARN] Skipping malformed filename: {mask_path}")
        return []

    tree_records = []
    
    try:
        with rasterio.open(mask_path) as src:
            # Read entire 100km mask into memory (uint8)
            # 10,000 x 10,000 x 1 byte = 100 MB RAM (very safe)
            data = src.read(1)
            
            # Iterate through 5x5 grid of 20km subtiles
            for r_idx in range(GRID_N):
                for c_idx in range(GRID_N):
                    # Construct subtile ID matching the downloader convention
                    subtile_id = f"{base_name}_sr{r_idx}_sc{c_idx}"
                    
                    # Optimization: Skip extraction if this subtile isn't in our list
                    if valid_subtiles is not None and subtile_id not in valid_subtiles:
                        continue

                    # Calculate window coordinates
                    r_start = r_idx * SUBTILE_SIZE
                    r_end = r_start + SUBTILE_SIZE
                    c_start = c_idx * SUBTILE_SIZE
                    c_end = c_start + SUBTILE_SIZE
                    
                    # Extract subtile
                    subtile = data[r_start:r_end, c_start:c_end]
                    
                    # Find tree pixels (value == 1)
                    # np.where returns (row_indices, col_indices)
                    rows, cols = np.where(subtile > 0)
                    
                    if len(rows) > 0:
                        # Create record for this subtile
                        # We store arrays directly to create DataFrame later
                        # to avoid creating millions of dict objects
                        df_chunk = pd.DataFrame({
                            'tile_id': subtile_id,
                            'row': rows.astype(np.uint16),
                            'col': cols.astype(np.uint16)
                        })
                        tree_records.append(df_chunk)
                        
    except Exception as e:
        print(f"[ERROR] Failed processing {mask_path}: {e}")
        return []

    return tree_records

def main():
    parser = argparse.ArgumentParser(description="Generate global tree pixel index from masks.")
    parser.add_argument("--mask_dir", required=True, help="Directory containing 100km .tif masks")
    parser.add_argument("--output_dir", required=True, help="Output directory for Parquet shards")
    parser.add_argument("--completed_tiles_file", required=False, help="Path to text file containing list of completed 20km tile IDs to filter by")
    parser.add_argument("--shard_rows", type=int, default=1_000_000, help="Approx rows per output file")
    
    args = parser.parse_args()
    
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optional: Load completed tiles whitelist
    valid_subtiles = None
    if args.completed_tiles_file:
        p = Path(args.completed_tiles_file)
        if p.exists():
            with open(p, 'r') as f:
                # Read lines, strip whitespace, remove empty lines
                valid_subtiles = set(line.strip() for line in f if line.strip())
            print(f"Filtering index to {len(valid_subtiles)} completed tiles from {p.name}")
        else:
            print(f"[WARN] Completed tiles file not found: {p}. Proceeding with ALL tiles.")

    # 1. Find all masks
    mask_files = list(mask_dir.glob("*.tif"))
    print(f"Found {len(mask_files)} mask files.")
    
    # 2. Process in batches to manage memory
    # We accumulate records in memory until we hit a threshold, then shuffle and write.
    
    current_records = []
    total_pixels_count = 0
    shard_counter = 0
    
    print("Processing masks and building index...")
    for f in tqdm(mask_files):
        chunks = process_single_mask(f, valid_subtiles)
        if chunks:
            current_records.extend(chunks)
            
            # Check size (rough estimate)
            # Sum of lengths of all dataframes in list
            current_len = sum(len(df) for df in current_records)
            
            if current_len >= args.shard_rows:
                # 3. Merge, Shuffle, Write Shard
                full_df = pd.concat(current_records, ignore_index=True)
                
                # Add random sort key
                full_df['rand_sort'] = np.random.random(size=len(full_df)).astype(np.float32)
                
                # Sort by random key
                full_df = full_df.sort_values('rand_sort')
                
                # Write to Parquet
                out_path = output_dir / f"part-{shard_counter:05d}.parquet"
                full_df.to_parquet(out_path, index=False, compression='snappy')
                
                print(f"  --> Wrote {len(full_df):,} pixels to {out_path.name}")
                
                total_pixels_count += len(full_df)
                shard_counter += 1
                current_records = [] # Reset buffer

    # 4. Write remaining records
    if current_records:
        full_df = pd.concat(current_records, ignore_index=True)
        full_df['rand_sort'] = np.random.random(size=len(full_df)).astype(np.float32)
        full_df = full_df.sort_values('rand_sort')
        
        out_path = output_dir / f"part-{shard_counter:05d}.parquet"
        full_df.to_parquet(out_path, index=False, compression='snappy')
        print(f"  --> Wrote {len(full_df):,} pixels to {out_path.name} (Final)")
        total_pixels_count += len(full_df)

    print("="*40)
    print(f"Done. Total Tree Pixels Indexed: {total_pixels_count:,}")
    print(f"Index shards saved to: {output_dir}")

if __name__ == "__main__":
    main()

