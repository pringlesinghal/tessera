#!/usr/bin/env python3
"""
explore_tree_mask.py

Loads a tree mask GeoTIFF and performs exploratory analysis:
- Prints metadata (CRS, dimensions, transform)
- Calculates statistics (min, max, mean)
- Evaluates as binary mask (>0) to report tree pixel counts and sparsity
"""

import argparse
import sys
from pathlib import Path
import rasterio
import numpy as np

# Default location on Sherlock
DEFAULT_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_masks"

def explore_mask(file_path):
    print(f"\nAnalyzing: {file_path}")
    print("=" * 60)
    
    try:
        with rasterio.open(file_path) as src:
            # 1. Metadata
            print(f"Driver:    {src.driver}")
            print(f"Size:      {src.width} x {src.height}")
            print(f"Bands:     {src.count}")
            print(f"CRS:       {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Dtype:     {src.dtypes[0]}")
            print(f"Bounds:    {src.bounds}")
            
            # 2. Load Data (Band 1)
            # Use masked=True if nodata is defined, otherwise standard read
            data = src.read(1, masked=True)
            
            # Handle MaskedArray if returned
            if np.ma.is_masked(data):
                valid_data = data.compressed()
                total_pixels = data.size
            else:
                valid_data = data.flatten()
                total_pixels = data.size

            # 3. Basic Statistics
            print("\n--- Data Statistics ---")
            print(f"Min:  {valid_data.min()}")
            print(f"Max:  {valid_data.max()}")
            print(f"Mean: {valid_data.mean():.4f}")
            
            unique_vals = np.unique(valid_data)
            if len(unique_vals) < 20:
                print(f"Unique values: {unique_vals}")
            else:
                print(f"Unique values count: {len(unique_vals)}")
                print(f"First 10 unique values: {unique_vals[:10]} ...")

            # 4. Binary Analysis (Assuming > 0 is tree)
            print("\n--- Binary Mask Analysis (value > 0) ---")
            
            # Count pixels > 0 (treating Nodata as 0 if not handled by mask)
            if np.ma.is_masked(data):
                binary_mask = data > 0
                tree_pixels = binary_mask.sum()
            else:
                binary_mask = data > 0
                tree_pixels = np.sum(binary_mask)
            
            coverage_pct = (tree_pixels / total_pixels) * 100
            
            print(f"Total Pixels: {total_pixels:,}")
            print(f"Tree Pixels:  {tree_pixels:,}")
            print(f"Tree Cover:   {coverage_pct:.4f}%")
            
            # 5. Index Estimation
            if tree_pixels > 0:
                # coordinate indices (row, col) as int32 would be 8 bytes per pixel
                mem_indices_mb = (tree_pixels * 8) / (1024 * 1024)
                print(f"\n--- Storage Estimation ---")
                print(f"Est. memory for (row, col) int32 indices: {mem_indices_mb:.2f} MB")
                print(f"Est. memory for (row, col, time, 12_bands) uint16 (Time=30): {(tree_pixels * 30 * 12 * 2) / (1024**3):.2f} GB raw data")

    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Explore tree mask TIFFs.")
    parser.add_argument("path", nargs="?", default=DEFAULT_DIR, 
                        help=f"Path to TIF file or directory (default: {DEFAULT_DIR})")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        # Try checking if it is a relative path in the container structure
        # or warn user
        print(f"[WARNING] Path not found locally: {path}")
        print("Ensure you are running this script on the cluster node where data exists.")
        sys.exit(1)
    
    if path.is_file():
        explore_mask(path)
    elif path.is_dir():
        print(f"Searching for .tif files in {path}...")
        tifs = list(path.glob("*.tif"))
        if not tifs:
            print(f"No .tif files found in {path}")
            sys.exit(1)
        
        print(f"Found {len(tifs)} TIF files.")
        
        # Pick a sample: try to find a 'zone' file if mixed, or just the first one
        sample = next((f for f in tifs if "zone" in f.name), tifs[0])
        explore_mask(sample)
    else:
        print(f"Invalid path type: {path}")
        sys.exit(1)

if __name__ == "__main__":
    main()

