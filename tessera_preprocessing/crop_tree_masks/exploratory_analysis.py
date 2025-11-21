import os
import numpy as np
from pathlib import Path
import random

FARMTREE_TILE_DIR = Path("/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/farmtree_tiles")
SENTINEL_DATA_DIR = Path("/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/complete_time_series_test/downloaded_tiles")

YEARS = list(range(2016, 2025))  # 2016–2024 inclusive


def find_data_processed(tile_id):
    """
    Returns: list of (year, data_processed_path) for all years that exist.
    """
    found = []
    for year in YEARS:
        p = SENTINEL_DATA_DIR / tile_id / str(year) / "data_processed"
        if p.exists():
            found.append((year, p))
    return found

YEARS = list(range(2016, 2025))

def sample_missing_fraction(bands, n_samples=20000):
    """Estimate missingness without scanning entire array."""
    T, H, W, C = bands.shape
    samples = 0
    nan_count = 0

    for _ in range(n_samples):
        t = random.randrange(T)
        r = random.randrange(H)
        c = random.randrange(W)
        if np.isnan(bands[t, r, c, 0]):
            nan_count += 1
        samples += 1

    return nan_count / samples


def summarize_cube(dp_dir):
    bands = np.load(dp_dir / "bands.npy", mmap_mode="r")
    doys = np.load(dp_dir / "doys.npy")
    mask = np.load(dp_dir / "masks.npy", mmap_mode="r")  # LARGE – mmap required

    T, H, W, C = bands.shape

    print(f"  Cube shape          : T={T}, H={H}, W={W}, C={C}")
    print(f"  DOY range           : {doys.min()} → {doys.max()}")
    print(f"  Unique DOYs         : {len(np.unique(doys))}")

    # Approximate missing fraction safely
    missing_est = sample_missing_fraction(bands)
    print(f"  Missing fraction    : ~{missing_est:.4f} (sampled)")

    print(f"  Mask shape          : {mask.shape}")

    # If mask is per-timestep, collapse it
    if mask.ndim == 3:
        print("  Mask is time-varying → collapsing via OR.")
        mask_static = (mask.sum(axis=0) > 0)
    else:
        mask_static = mask == 1

    print(f"  Mask sparsity       : {(mask_static).mean():.4f}")

    # Temporal gap statistics
    diffs = np.diff(np.sort(doys))
    if len(diffs) > 0:
        print(f"  ΔDOY min/median/max : {diffs.min()} / {np.median(diffs)} / {diffs.max()}")
    else:
        print("  ΔDOY stats          : Not enough observations")

    # Sample a pixel time series
    coords = list(zip(*np.where(mask_static)))
    if len(coords) > 0:
        r, c = random.choice(coords)
        ts = bands[:, r, c, :]
        valid = ~np.isnan(ts[:, 0])

        print(f"  Sample pixel (r={r}, c={c})")
        print(f"    Valid timesteps   : {valid.sum()}/{len(valid)}")
        print(f"    DOYs (first 10)   : {doys[valid][:10]}")
        print(f"    First band sample : {ts[valid, 0][:10]}")
    else:
        print("  No masked pixels to sample.")


import rasterio

def summarize_farmtree_mask(tile_path):
    """Load farmtree .tif (uint8) and compute fraction of pixels == 1."""
    with rasterio.open(tile_path) as ds:
        data = ds.read(1)  # read single band
    total = data.size
    ones = (data == 1).sum()
    fraction = ones / total
    return total, ones, fraction


def main():
    tile_files = sorted(FARMTREE_TILE_DIR.glob("*.tif"))
    print(tile_files)
    tile_ids = [f.stem.replace("tile_", "") for f in tile_files]

    print(f"Found {len(tile_ids)} farmtree tiles.")

    for tile_file, tile_id in zip(tile_files, tile_ids):
        print("\n" + "="*80)
        print(f" TILE: {tile_id}")
        print("="*80)

        # ---- NEW: Load farmtree mask and compute sparsity ----
        total, ones, frac = summarize_farmtree_mask(tile_file)
        print(f"  FarmTree mask: {ones}/{total} pixels = {frac*100:.2f}% == 1")

        # ---- existing Sentinel search ----
        matches = find_data_processed(tile_id)
        if not matches:
            print("  No Sentinel data found for this tile.")
            continue


    for tile_id in tile_ids:
        print("\n" + "="*80)
        print(f" TILE: {tile_id}")
        print("="*80)

        matches = find_data_processed(tile_id)
        if not matches:
            print("  No Sentinel data found for this tile.")
            continue

        for year, dp_dir in matches:
            print(f"\n--- YEAR {year} ---")
            try:
                summarize_cube(dp_dir)
            except:
                print(f"missing data {tile_id=},{year=},{dp_dir=}")


if __name__ == "__main__":
    main()