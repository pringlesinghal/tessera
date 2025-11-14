#!/usr/bin/env python3
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import fiona
from shapely.geometry import shape, box
from shapely.ops import unary_union, transform as shp_transform
from shapely import wkb
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.crs import CRS
from pyproj import Transformer
import numpy as np
import gc
import math

# ---------------------------
# CONFIG
# ---------------------------
SHAPEFILE = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/reprojected_india.shp"
OUTPUT_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/tiles_utm"
SUB_TILE_SIZE_M = 20000  # 20 km sub-tile
MAX_WORKERS = 4
BATCH_SIZE = 50
LOG_EVERY_N = 50
PIXEL_SIZE = 10
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mgrs_tiling")

# ---------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------
def create_subtile_windows(min_e, min_n, max_e, max_n, tile_size):
    windows = []
    n_rows = int(math.ceil((max_n - min_n) / tile_size))
    n_cols = int(math.ceil((max_e - min_e) / tile_size))
    for r in range(n_rows):
        for c in range(n_cols):
            e0 = min_e + c * tile_size
            n0 = min_n + r * tile_size
            e1 = min(e0 + tile_size, max_e)
            n1 = min(n0 + tile_size, max_n)
            windows.append((r, c, e0, n0, e1, n1))
    return windows

def worker_rasterize_subtile(subtile_wkb, e0, n0, e1, n1, pixel_size, epsg, r, c, out_dir):
    subtile_geom = wkb.loads(subtile_wkb)
    tile_box = box(e0, n0, e1, n1)
    if not subtile_geom.intersects(tile_box):
        return None

    clipped = subtile_geom.intersection(tile_box)
    if clipped.is_empty:
        return None

    width = int(np.ceil((e1 - e0) / pixel_size))
    height = int(np.ceil((n1 - n0) / pixel_size))
    transform_affine = from_origin(e0, n1, pixel_size, pixel_size)
    raster = rasterize([(clipped, 255)], out_shape=(height, width), transform=transform_affine, fill=0, dtype='uint8')

    filename = os.path.join(out_dir, f"tile_zone{epsg}_r{r}_c{c}.tif")
    with rasterio.open(
        filename, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=CRS.from_epsg(epsg),
        transform=transform_affine,
        compress="LZW"
    ) as dst:
        dst.write(raster, 1)

    return filename

# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------
def generate_india_tiles(shp_path, output_dir, sub_tile_size=SUB_TILE_SIZE_M,
                         max_workers=MAX_WORKERS, batch_size=BATCH_SIZE, pixel_size=PIXEL_SIZE):
    logger.info("Opening shapefile: %s", shp_path)
    with fiona.open(shp_path, "r") as src:
        geoms = [shape(feat["geometry"]) for feat in src]
        src_crs = CRS(src.crs) if src.crs else CRS.from_epsg(4326)
    union_geom = unary_union(geoms)

    # Transform to WGS84 if needed
    if src_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True).transform
        union_wgs84 = shp_transform(transformer, union_geom)
    else:
        union_wgs84 = union_geom

    min_lon, min_lat, max_lon, max_lat = union_wgs84.bounds
    logger.info("India bounding box: %.3f, %.3f -> %.3f, %.3f", min_lon, min_lat, max_lon, max_lat)

    # Find UTM zones intersecting India
    min_zone = int((min_lon + 180) / 6) + 1
    max_zone = int((max_lon + 180) / 6) + 1
    logger.info("UTM zones intersecting India: %d - %d", min_zone, max_zone)

    os.makedirs(output_dir, exist_ok=True)
    all_tiles = []

    for zone in range(min_zone, max_zone + 1):
        epsg = 32600 + zone  # Northern hemisphere

        # Clip India to current UTM zone
        zone_lon_min = (zone - 1) * 6 - 180
        zone_lon_max = zone * 6 - 180
        india_zone_geom = union_wgs84.intersection(box(zone_lon_min, -90, zone_lon_max, 90))
        if india_zone_geom.is_empty:
            continue

        # Transform clipped geometry to UTM
        transformer_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
        india_utm = shp_transform(transformer_to_utm, india_zone_geom)
        subtile_wkb = india_utm.wkb

        # Create subtiles
        min_e, min_n, max_e, max_n = india_utm.bounds
        windows = create_subtile_windows(min_e, min_n, max_e, max_n, sub_tile_size)
        total_windows = len(windows)
        logger.info("Zone %d: Total subtiles = %d", zone, total_windows)

        processed = 0
        for i in range(0, total_windows, batch_size):
            batch = windows[i:i + batch_size]
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {
                    exe.submit(worker_rasterize_subtile, subtile_wkb, e0, n0, e1, n1, pixel_size, epsg, r, c, output_dir): (r, c)
                    for (r, c, e0, n0, e1, n1) in batch
                }
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        all_tiles.append(res)
                    processed += 1
                    if processed % LOG_EVERY_N == 0:
                        pct = 100.0 * processed / total_windows
                        logger.info("Progress %.2f%% (%d/%d)", pct, processed, total_windows)

        gc.collect()

    logger.info("✅ Done. Generated %d tiles", len(all_tiles))
    return all_tiles

# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    tiles = generate_india_tiles(SHAPEFILE, OUTPUT_DIR)
    logger.info("Sample generated tiles: %s", tiles[:5])

if __name__ == "__main__":
    main()
