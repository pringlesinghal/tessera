import rasterio
from pathlib import Path

TILE_DIR = Path("/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/tiles_utm")

def load_and_print_tile(tile_id):
    # The tile filename format is likely:   tile_<tile_id>.tif
    # Example:  tile_zone32642_r19_c14.tif
    tile_path = TILE_DIR / f"tile_{tile_id}.tif"

    if not tile_path.exists():
        print(f"Tile not found: {tile_path}")
        return

    print(f"\n--- Loading {tile_path} ---")
    with rasterio.open(tile_path) as ds:
        print(f"Width  : {ds.width}")
        print(f"Height : {ds.height}")
        print(f"CRS    : {ds.crs}")
        print(f"Transform:\n{ds.transform}")

        # Optional: data type + nodata
        print(f"Dtype  : {ds.dtypes}")
        print(f"NoData : {ds.nodata}")


# Example usage
tile_ids = [
    "zone32642_r19_c14",
    "zone32642_r18_c4",
    "zone32642_r18_c9",
    "zone32642_r13_c15",
    "zone32642_r10_c10",
]

for tid in tile_ids:
    load_and_print_tile(tid)
