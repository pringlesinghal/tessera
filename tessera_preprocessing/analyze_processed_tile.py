#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np


def describe_npy(path: Path) -> None:
    if not path.exists():
        print(f"{path.name:<24} MISSING")
        return
    arr = np.load(path, mmap_mode="r")
    size_mb = path.stat().st_size / (1024 * 1024)
    stats = {
        "shape": arr.shape,
        "dtype": arr.dtype,
        "size_mb": f"{size_mb:.2f}",
    }
    sample = arr.flat[0] if arr.size > 0 else "NA"
    print(f"{path.name:<24} shape={stats['shape']}, dtype={stats['dtype']}, sizeMB={stats['size_mb']}, sample={sample}")


def describe_text(path: Path) -> None:
    if not path.exists():
        print(f"{path.name:<24} MISSING")
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"{path.name:<24} text file, sizeMB={size_mb:.2f}")
    with path.open() as f:
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            print(f"  {line.strip()}")


def main(processed_dir: Path) -> None:
    files = [
        "bands.npy",
        "doys.npy",
        "band_mean.npy",
        "band_std.npy",
        "masks.npy",
        "sar_ascending.npy",
        "sar_ascending_doy.npy",
        "sar_descending.npy",
        "sar_descending_doy.npy",
    ]
    print(f"Inspecting {processed_dir}")
    for fname in files:
        describe_npy(processed_dir / fname)
    describe_text(processed_dir / "processing_info.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_processed_tile.py /path/to/data_processed")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve())

