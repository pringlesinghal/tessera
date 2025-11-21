#!/usr/bin/env python3
"""
Generate MGRS-aligned GeoTIFF tiles and manifests for both mask (100 km) and
downloader (20 km) workflows.

The script creates two raster products:
  1. **Parent 100 km × 100 km tiles**: each aligned to the native Sentinel 10 m
     grid. These are the canonical inputs for Earth Engine farm-tree mask jobs.
  2. **20 km × 20 km subtiles**: the per-job Tessera downloader inputs.

Both products are clipped to the ROI shapefile, rasterized at 10 m, and written
alongside manifests (CSV + tile lists) capturing IDs, UTM metadata, and coverage
statistics so downstream pipelines can stay in sync.
"""

import argparse
import csv
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import fiona
import numpy as np
import rasterio
from geographiclib.mgrs import MGRS
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

LOGGER = logging.getLogger("mgrs_tiler")
MGRS_HELPER = MGRS()


@dataclass
class ParentTileMetadata:
    tile_id: str
    mgrs_100km: str
    zone: int
    epsg: int
    row: int
    col: int
    min_e: float
    min_n: float
    max_e: float
    max_n: float
    centroid_lat: float
    centroid_lon: float
    coverage_ratio: float


@dataclass
class SubtileMetadata:
    tile_id: str
    parent_tile_id: str
    mgrs_100km: str
    zone: int
    epsg: int
    row: int
    col: int
    min_e: float
    min_n: float
    max_e: float
    max_n: float
    centroid_lat: float
    centroid_lon: float
    coverage_ratio: float


def _read_union_geometry(shapefile: Path):
    LOGGER.info("Reading shapefile: %s", shapefile)
    with fiona.open(shapefile, "r") as src:
        input_crs = src.crs
        geoms = [shape(feat["geometry"]) for feat in src]

    if not geoms:
        raise ValueError("Input shapefile contains no geometries.")

    union_geom = unary_union(geoms)

    if not input_crs or input_crs.get("init", "").lower() == "epsg:4326":
        LOGGER.info("Shapefile already in WGS84.")
        return union_geom

    LOGGER.info("Reprojecting shapefile from %s to EPSG:4326.", input_crs)
    transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
    return shp_transform(transformer.transform, union_geom)


def _utm_epsg(zone: int, northern: bool) -> int:
    return (32600 if northern else 32700) + zone


def _zone_lon_bounds(zone: int) -> Tuple[float, float]:
    min_lon = (zone - 1) * 6 - 180
    max_lon = zone * 6 - 180
    return min_lon, max_lon


def _iter_zone_parts(geom_wgs84, zone: int) -> Iterable[Tuple[int, bool, object]]:
    lon_min, lon_max = _zone_lon_bounds(zone)
    zone_strip = geom_wgs84.intersection(box(lon_min, -90, lon_max, 90))
    if zone_strip.is_empty:
        return

    north = zone_strip.intersection(box(lon_min, 0, lon_max, 90))
    south = zone_strip.intersection(box(lon_min, -90, lon_max, 0))

    if not north.is_empty:
        yield zone, True, north
    if not south.is_empty:
        yield zone, False, south


def _snap_bounds(min_coord: float, max_coord: float, tile_size: int) -> Tuple[int, int]:
    snapped_min = int(math.floor(min_coord / tile_size) * tile_size)
    snapped_max = int(math.ceil(max_coord / tile_size) * tile_size)
    return snapped_min, snapped_max


def _mgrs_id_from_centroid(lon: float, lat: float) -> str:
    return MGRS_HELPER.Forward(lat, lon, 0)


def _write_manifest(manifest_path: Path, records: Sequence[object]):
    if not records:
        LOGGER.warning("No records to write at %s", manifest_path)
        return

    fieldnames = list(asdict(records[0]).keys())
    LOGGER.info("Writing manifest with %d entries to %s", len(records), manifest_path)
    with manifest_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_tile_list(tile_list_path: Path, records: Sequence[object]):
    LOGGER.info("Writing tile list (%d entries) to %s", len(records), tile_list_path)
    with tile_list_path.open("w") as f:
        for record in records:
            f.write(f"{getattr(record, 'tile_id')}\n")


def generate_mgrs_tiles(
    shapefile: Path,
    output_dir: Path,
    tile_size: int = 100_000,
    subtile_size: int = 20_000,
    pixel_size: float = 10.0,
    overwrite: bool = False,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_dir = output_dir / "tiles_100km"
    subtile_dir = output_dir / "tiles_20km"
    parent_dir.mkdir(parents=True, exist_ok=True)
    subtile_dir.mkdir(parents=True, exist_ok=True)

    union_geom = _read_union_geometry(shapefile)
    min_lon, _, max_lon, _ = union_geom.bounds
    min_zone = max(1, int(math.floor((min_lon + 180) / 6) + 1))
    max_zone = min(60, int(math.ceil((max_lon + 180) / 6)))

    parent_records: List[ParentTileMetadata] = []
    subtile_records: List[SubtileMetadata] = []

    for zone in range(min_zone, max_zone + 1):
        for zone_num, northern, zone_geom in _iter_zone_parts(union_geom, zone):
            epsg = _utm_epsg(zone_num, northern)
            LOGGER.info(
                "Processing zone %d (%s hemisphere) -> EPSG:%d",
                zone_num,
                "N" if northern else "S",
                epsg,
            )
            transformer_to_utm = Transformer.from_crs(
                "EPSG:4326", f"EPSG:{epsg}", always_xy=True
            )
            transformer_to_wgs84 = Transformer.from_crs(
                f"EPSG:{epsg}", "EPSG:4326", always_xy=True
            )
            utm_geom = shp_transform(transformer_to_utm.transform, zone_geom)
            if utm_geom.is_empty:
                continue

            min_x, min_y, max_x, max_y = utm_geom.bounds
            start_e, end_e = _snap_bounds(min_x, max_x, tile_size)
            start_n, end_n = _snap_bounds(min_y, max_y, tile_size)

            for tile_e in range(start_e, end_e, tile_size):
                for tile_n in range(start_n, end_n, tile_size):
                    utm_tile = box(
                        tile_e, tile_n, tile_e + tile_size, tile_n + tile_size
                    )
                    tile_intersection = utm_geom.intersection(utm_tile)
                    if tile_intersection.is_empty:
                        continue

                    tile_row_idx = int(math.floor(tile_n / tile_size))
                    tile_col_idx = int(math.floor(tile_e / tile_size))
                    tile_center_lon, tile_center_lat = transformer_to_wgs84.transform(
                        tile_e + tile_size / 2, tile_n + tile_size / 2
                    )
                    parent_mgrs = _mgrs_id_from_centroid(
                        tile_center_lon, tile_center_lat
                    )
                    parent_tile_id = f"zone{epsg}_r{tile_row_idx}_c{tile_col_idx}"
                    parent_tile_path = parent_dir / f"tile_{parent_tile_id}.tif"
                    parent_transform = from_origin(
                        tile_e, tile_n + tile_size, pixel_size, pixel_size
                    )
                    parent_dim = int(round(tile_size / pixel_size))

                    if parent_dim <= 0:
                        raise ValueError("Invalid parent tile dimension computed.")

                    if not parent_tile_path.exists() or overwrite:
                        parent_mask = rasterize(
                            [(tile_intersection, 1)],
                            out_shape=(parent_dim, parent_dim),
                            transform=parent_transform,
                            fill=0,
                            dtype="uint8",
                        )
                        LOGGER.debug("Writing parent tile %s", parent_tile_path)
                        with rasterio.open(
                            parent_tile_path,
                            "w",
                            driver="GTiff",
                            height=parent_dim,
                            width=parent_dim,
                            count=1,
                            dtype=parent_mask.dtype,
                            transform=parent_transform,
                            crs=f"EPSG:{epsg}",
                            compress="LZW",
                        ) as dst_parent:
                            dst_parent.write(parent_mask, 1)
                    else:
                        LOGGER.debug(
                            "Parent tile %s exists; skipping write (use --overwrite).",
                            parent_tile_path,
                        )

                    parent_centroid = tile_intersection.centroid
                    centroid_lon, centroid_lat = transformer_to_wgs84.transform(
                        parent_centroid.x, parent_centroid.y
                    )
                    parent_coverage = tile_intersection.area / (tile_size * tile_size)
                    parent_records.append(
                        ParentTileMetadata(
                            tile_id=parent_tile_id,
                            mgrs_100km=parent_mgrs,
                            zone=zone_num,
                            epsg=epsg,
                            row=tile_row_idx,
                            col=tile_col_idx,
                            min_e=tile_e,
                            min_n=tile_n,
                            max_e=tile_e + tile_size,
                            max_n=tile_n + tile_size,
                            centroid_lat=centroid_lat,
                            centroid_lon=centroid_lon,
                            coverage_ratio=parent_coverage,
                        )
                    )

                    for row_idx, sub_n in enumerate(
                        range(tile_n, tile_n + tile_size, subtile_size)
                    ):
                        for col_idx, sub_e in enumerate(
                            range(tile_e, tile_e + tile_size, subtile_size)
                        ):
                            subtile = box(
                                sub_e, sub_n, sub_e + subtile_size, sub_n + subtile_size
                            )
                            clipped = tile_intersection.intersection(subtile)
                            if clipped.is_empty:
                                continue

                            transform = from_origin(
                                sub_e, sub_n + subtile_size, pixel_size, pixel_size
                            )
                            width = height = int(round(subtile_size / pixel_size))
                            mask = rasterize(
                                [(clipped, 1)],
                                out_shape=(height, width),
                                transform=transform,
                                fill=0,
                                dtype="uint8",
                            )

                            if not mask.any():
                                continue

                            centroid = clipped.centroid
                            lon, lat = transformer_to_wgs84.transform(
                                centroid.x, centroid.y
                            )
                            mgrs_100km = _mgrs_id_from_centroid(lon, lat)
                            subtile_id = f"{mgrs_100km}_r{row_idx}_c{col_idx}"

                            tile_path = subtile_dir / f"tile_{subtile_id}.tif"
                            if tile_path.exists() and not overwrite:
                                LOGGER.warning(
                                    "Tile %s exists; skipping (use --overwrite).",
                                    tile_path,
                                )
                                continue

                            LOGGER.debug("Writing tile %s", tile_path)
                            with rasterio.open(
                                tile_path,
                                "w",
                                driver="GTiff",
                                height=height,
                                width=width,
                                count=1,
                                dtype=mask.dtype,
                                transform=transform,
                                crs=f"EPSG:{epsg}",
                                compress="LZW",
                            ) as dst:
                                dst.write(mask, 1)

                            coverage = clipped.area / (subtile_size * subtile_size)
                            subtile_records.append(
                                SubtileMetadata(
                                    tile_id=subtile_id,
                                    parent_tile_id=parent_tile_id,
                                    mgrs_100km=mgrs_100km,
                                    zone=zone_num,
                                    epsg=epsg,
                                    row=row_idx,
                                    col=col_idx,
                                    min_e=sub_e,
                                    min_n=sub_n,
                                    max_e=sub_e + subtile_size,
                                    max_n=sub_n + subtile_size,
                                    centroid_lat=lat,
                                    centroid_lon=lon,
                                    coverage_ratio=coverage,
                                )
                            )

    parent_manifest_path = output_dir / "parent_tile_manifest.csv"
    subtile_manifest_path = output_dir / "subtile_manifest.csv"
    parent_tile_list_path = output_dir / "parent_tile_list.txt"
    subtile_tile_list_path = output_dir / "subtile_tile_list.txt"

    _write_manifest(parent_manifest_path, parent_records)
    _write_manifest(subtile_manifest_path, subtile_records)
    _write_tile_list(parent_tile_list_path, parent_records)
    _write_tile_list(subtile_tile_list_path, subtile_records)
    LOGGER.info(
        "Finished generating %d parent tiles and %d subtiles.",
        len(parent_records),
        len(subtile_records),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MGRS-aligned GeoTIFF tiles.")
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=Path(
            "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/reprojected_india.shp"
        ),
        help="Path to the ROI shapefile (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles"
        ),
        help="Directory for generated tiles and manifests (default: %(default)s).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=100_000,
        help="Parent tile size in meters (100 km MGRS grid).",
    )
    parser.add_argument(
        "--subtile-size",
        type=int,
        default=20_000,
        help="Subtile size in meters (default: 20 km).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=10.0,
        help="Pixel size in meters for rasterization (default: 10 m).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GeoTIFFs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    generate_mgrs_tiles(
        shapefile=args.shapefile,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        subtile_size=args.subtile_size,
        pixel_size=args.pixel_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
