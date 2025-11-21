import os
import numpy as np
import rasterio
import zarr

# ------------------------
# Config
# ------------------------
TILE_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/complete_time_series_test/downloaded_tiles"
FARM_MASK_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/farmtree_tiles"
OUTPUT_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/zarr_output"

CHUNK_PIXELS = 10000
CHUNK_TIME = 12

SENSORS = ["S2", "S1A", "S1D"]

# ------------------------
# Helper functions
# ------------------------

def load_farm_mask(tile_id):
    path = os.path.join(FARM_MASK_DIR, f"{tile_id}.tif")
    mask = rasterio.open(path).read(1)
    # Trim if not divisible
    if mask.shape[0] != mask.shape[1]:
        print(f"  ⚠ Farm mask shape {mask.shape} → trimming to {min(mask.shape)}x{min(mask.shape)}")
        mask = mask[:min(mask.shape), :min(mask.shape)]
    return mask.astype(bool)

def load_sensor_cube(tile_id, year, sensor):
    base_path = os.path.join(TILE_DIR, tile_id, str(year), "data_processed")
    
    try:
        if sensor == "S2":
            cube = np.load(os.path.join(base_path, "bands.npy"))  # (T,H,W,C)
            doys = np.load(os.path.join(base_path, "doys.npy"))
        elif sensor == "S1A":
            cube = np.load(os.path.join(base_path, "sar_ascending.npy"))
            doys = np.load(os.path.join(base_path, "sar_ascending_doy.npy"))
        elif sensor == "S1D":
            cube = np.load(os.path.join(base_path, "sar_descending.npy"))
            doys = np.load(os.path.join(base_path, "sar_descending_doy.npy"))
        else:
            raise ValueError(f"Unknown sensor {sensor}")
        mask = np.load(os.path.join(base_path, "masks.npy"))  # (T,H,W)
        
        # Ensure cube and mask shapes align
        if cube.shape[:2] != mask.shape[:2]:
            print(f"  ⚠ Cube and mask spatial shape mismatch: cube {cube.shape[:2]}, mask {mask.shape[:2]}")
            min_h = min(cube.shape[0], mask.shape[0])
            min_w = min(cube.shape[1], mask.shape[1])
            cube = cube[:min_h, :min_w, ...]
            mask = mask[:min_h, :min_w, ...]
        
        if cube.shape[0] != mask.shape[0]:
            print(f"  ⚠ Cube and mask time dimension mismatch: cube T={cube.shape[0]}, mask T={mask.shape[0]}")
            T = min(cube.shape[0], mask.shape[0])
            cube = cube[:T]
            mask = mask[:T]
        
        return cube, doys, mask
    except FileNotFoundError:
        print(f"   ⚠ No valid data for {sensor}")
        return None, None, None

def apply_masks(cube, farm_mask, sensor_mask):
    """Return pixels of shape (N_pixels, T, C) with invalid entries as NaN"""
    if cube is None:
        return None, None, None
    T, H, W = cube.shape[:3]
    C = cube.shape[3] if cube.ndim == 4 else 1

    # Flatten spatial
    rows, cols = np.where(farm_mask)
    N_pixels = len(rows)
    print(f"  Farm-tree pixels: {N_pixels}")

    if N_pixels == 0:
        return None, None, None

    pixels = np.full((N_pixels, T, C), np.nan, dtype="float32")

    for t in range(T):
        valid = sensor_mask[t, rows, cols]
        if C > 1:
            pixels[valid, t, :] = cube[t, rows[valid], cols[valid], :]
        else:
            pixels[valid, t, 0] = cube[t, rows[valid], cols[valid]]
    return pixels, rows, cols

def write_zarr(sensor, tile_id, year, pixels, doys, rows, cols):
    out_path = os.path.join(OUTPUT_DIR, tile_id, f"{year}_{sensor}.zarr")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open(out_path, mode="w")

    if pixels is not None:
        N, T, C = pixels.shape
        print(f"  ✔ {sensor} stored: {N} pixels, T={T}, C={C}")
        root.create_array("timeseries", shape=(N, T, C), chunks=(min(CHUNK_PIXELS, N), min(CHUNK_TIME, T), C),
                          dtype="float32", overwrite=True)
        root["timeseries"][:] = pixels
        root.create_array("doys", data=doys.astype("int16"), overwrite=True)
        root.create_array("rows", data=rows, overwrite=True)
        root.create_array("cols", data=cols, overwrite=True)
    else:
        print(f"   ⚠ No valid pixels for {sensor}")

# ------------------------
# Main
# ------------------------

def process_tile(tile_id):
    print("="*80)
    print(f"TILE: {tile_id}")
    print("="*80)

    farm_mask = load_farm_mask(tile_id)
    years = [d for d in os.listdir(os.path.join(TILE_DIR, tile_id)) if d.isdigit()]
    years.sort()

    for year in years:
        print(f"\n  Year {year}")
        for sensor in SENSORS:
            cube, doys, sensor_mask = load_sensor_cube(tile_id, year, sensor)
            pixels, rows, cols = apply_masks(cube, farm_mask, sensor_mask)
            write_zarr(sensor, tile_id, year, pixels, doys, rows, cols)

def main():
    tile_ids = [f[:-4] for f in os.listdir(FARM_MASK_DIR) if f.endswith(".tif")]
    print(f"Found {len(tile_ids)} tiles with farmtree masks")
    for tile_id in tile_ids:
        process_tile(tile_id)

if __name__ == "__main__":
    main()
