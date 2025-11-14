#!/usr/bin/env python3
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window, bounds as window_bounds
from rasterio.crs import CRS
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform, unary_union
from shapely import wkb
from pyproj import Transformer
import numpy as np
import gc

# ---------------------------
# CONFIG
# ---------------------------
SHAPEFILE = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/reprojected_india.shp"
OUTPUT_TIFF = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/india_base.tif"
PIXEL_SIZE = 10
WINDOW_SIZE = 2048
MAX_WORKERS = 3            # memory-safe limit
BATCH_SIZE = 100           # process windows in small batches
LOG_EVERY_N = 50
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("shp2tiles")


def _create_window_list(width: int, height: int, window_size: int):
    windows = []
    for row_off in range(0, height, window_size):
        for col_off in range(0, width, window_size):
            win_h = min(window_size, height - row_off)
            win_w = min(window_size, width - col_off)
            windows.append((col_off, row_off, win_w, win_h))
    return windows


def determine_best_utm_crs_from_geom(geom_wgs84) -> CRS:
    lon, lat = geom_wgs84.centroid.x, geom_wgs84.centroid.y
    zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    epsg_code = 32600 + zone if is_northern else 32700 + zone
    logger.info("Picked UTM zone %d -> EPSG:%d (centroid lon=%.3f lat=%.3f)", zone, epsg_code, lon, lat)
    return CRS.from_epsg(epsg_code)


def worker_rasterize_window(union_wkb, minx, maxy, pixel_size, win_tuple):
    union_geom = wkb.loads(union_wkb)
    col_off, row_off, win_w, win_h = win_tuple
    transform_affine = from_origin(minx, maxy, pixel_size, pixel_size)
    win_minx, win_miny, win_maxx, win_maxy = window_bounds(Window(col_off, row_off, win_w, win_h), transform_affine)
    win_box = box(win_minx, win_miny, win_maxx, win_maxy)

    if not union_geom.intersects(win_box):
        return (col_off, row_off, win_w, win_h, None)

    clipped = union_geom.intersection(win_box)
    if clipped.is_empty:
        return (col_off, row_off, win_w, win_h, None)

    transform_win = rasterio.windows.transform(Window(col_off, row_off, win_w, win_h), transform_affine)
    mask = rasterize([(clipped, 255)], out_shape=(win_h, win_w), transform=transform_win, fill=0, dtype="uint8")
    return (col_off, row_off, win_w, win_h, mask)


def rasterize_india_to_tiles(
    shp_path: str,
    out_tiff: str,
    pixel_size: float = PIXEL_SIZE,
    window_size: int = WINDOW_SIZE,
    max_workers: int = MAX_WORKERS,
    batch_size: int = BATCH_SIZE,
):
    logger.info("Opening shapefile: %s", shp_path)
    with fiona.open(shp_path, "r") as src:
        src_crs = CRS(src.crs) if src.crs else CRS.from_epsg(4326)
        geoms = [shape(feat["geometry"]) for feat in src]
        logger.info("Read %d geometries", len(geoms))

    union_geom = unary_union(geoms)
    if src_crs.to_epsg() != 4326:
        to_wgs84 = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True).transform
        union_wgs84 = shp_transform(to_wgs84, union_geom)
    else:
        union_wgs84 = union_geom

    target_crs = determine_best_utm_crs_from_geom(union_wgs84)
    if src_crs != target_crs:
        to_target = Transformer.from_crs(src_crs, target_crs, always_xy=True).transform
        union_target = shp_transform(to_target, union_geom)
    else:
        union_target = union_geom
    union_wkb = union_target.wkb

    minx, miny, maxx, maxy = union_target.bounds
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    logger.info("Raster dimensions: %d x %d", width, height)

    tile_dir = os.path.join(os.path.dirname(out_tiff), "tiles")
    os.makedirs(tile_dir, exist_ok=True)

    windows = _create_window_list(width, height, window_size)
    total_windows = len(windows)
    logger.info("Total windows: %d", total_windows)

    processed = written = 0

    for i in range(0, total_windows, batch_size):
        batch = windows[i:i + batch_size]
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(worker_rasterize_window, union_wkb, minx, maxy, pixel_size, w): w
                for w in batch
            }

            for fut in as_completed(futures):
                win_tuple = futures[fut]
                try:
                    col_off, row_off, win_w, win_h, mask = fut.result()
                    if mask is None:
                        continue

                    win_transform = rasterio.windows.transform(
                        Window(col_off, row_off, win_w, win_h),
                        from_origin(minx, maxy, pixel_size, pixel_size)
                    )

                    tile_name = f"tile_r{row_off}_c{col_off}.tif"
                    tile_path = os.path.join(tile_dir, tile_name)

                    with rasterio.open(
                        tile_path, "w", driver="GTiff",
                        height=win_h, width=win_w, count=1,
                        dtype="uint8", crs=target_crs, transform=win_transform,
                        compress="LZW"
                    ) as dst_tile:
                        dst_tile.write(mask, 1)

                    written += 1
                except Exception as e:
                    logger.error("Worker failed: %s", e)
                processed += 1
                if processed % LOG_EVERY_N == 0:
                    pct = 100.0 * processed / total_windows
                    logger.info("Progress %.2f%% (%d/%d), written=%d",
                                pct, processed, total_windows, written)

        gc.collect()

    logger.info("✅ Done. Wrote %d GeoTIFF tiles to %s", written, tile_dir)
    return tile_dir


def main():
    rasterize_india_to_tiles(SHAPEFILE, OUTPUT_TIFF)


if __name__ == "__main__":
    main()

