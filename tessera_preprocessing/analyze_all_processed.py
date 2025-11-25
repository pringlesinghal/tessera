#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import numpy as np

def analyze_processed_folder(folder_path):
    """
    Scans a 'data_processed' folder, finds all .npy files,
    and prints their shape, dtype, and basic stats.
    """
    path = Path(folder_path)
    if not path.exists():
        print(f"[ERROR] Path does not exist: {path}")
        return

    print(f"\nAnalyzing Directory: {path}")
    print("=" * 60)

    # Find all .npy files
    npy_files = sorted(list(path.glob("*.npy")))
    
    if not npy_files:
        print("No .npy files found.")
        return

    for npy in npy_files:
        print(f"\nFile: {npy.name}")
        print("-" * 30)
        
        try:
            # Use mmap_mode='r' to avoid loading huge files into RAM just for shape
            data = np.load(npy, mmap_mode='r')
            
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Size:  {data.size * data.itemsize / (1024**2):.2f} MB")
            
            # If small enough, load to calculate stats (e.g. < 100MB)
            # Or if it's 1D, just load it
            is_small = (data.size * data.itemsize) < (100 * 1024 * 1024)
            is_1d = (data.ndim == 1)
            
            if is_small or is_1d:
                # Load fully for stats
                arr = np.array(data) 
                
                # Handle NaNs if float
                if np.issubdtype(arr.dtype, np.floating):
                    mn = np.nanmin(arr)
                    mx = np.nanmax(arr)
                    avg = np.nanmean(arr)
                    nans = np.isnan(arr).sum()
                    print(f"Min: {mn:.4f}, Max: {mx:.4f}, Mean: {avg:.4f}")
                    print(f"NaNs: {nans} ({nans/arr.size*100:.2f}%)")
                else:
                    print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.4f}")
            else:
                print("(File too large for full stats scan, showing first slice)")
                # Show corner value just to check validity
                if data.ndim == 4:
                    print(f"Sample [0,0,0]: {data[0,0,0]}")
                elif data.ndim == 3:
                    print(f"Sample [0,0,0]: {data[0,0,0]}")

        except Exception as e:
            print(f"[ERROR] Failed to read {npy.name}: {e}")

    # Check for txt files too
    txt_files = list(path.glob("*.txt"))
    for txt in txt_files:
        print(f"\nText File: {txt.name}")
        # Print first few lines
        with open(txt, 'r') as f:
            head = [next(f) for _ in range(5)]
        for line in head:
            print(f"  {line.strip()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to data_processed folder")
    args = parser.parse_args()
    
    analyze_processed_folder(args.path)

if __name__ == "__main__":
    main()

