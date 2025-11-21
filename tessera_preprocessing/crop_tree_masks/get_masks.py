"""
Farm tree mask builder with deterministic alignment to tessera downloader tiles.

Key features:
  * Enforces 10 m alignment using the exact affine transform stored in the local
    tessera tile GeoTIFF (so every exported pixel matches a downloader pixel).
  * Allows "small job" dry runs via --tiles/--max-tiles before launching a full batch.
  * Respects Earth Engine task limits by capping concurrent submissions.

Earth Engine projection control relies on `ee.Image.reproject` and exporting with
`crsTransform`, which, per the Earth Engine docs, gives full control over pixel
alignment (`ee.Image.reproject`: /wybert/earthengine-doc-md). Export tasks use
`Export.image.toCloudStorage` with a custom transform instead of `scale` so the
output grid exactly matches the input tile (see the CRS transform example explained
in `/wybert/earthengine-doc-md`'s `export-image-tocloudstorage.md`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ee
import rasterio
from rasterio.io import DatasetReader


# ---------------------------------------------------------------------------
# Defaults & constants
# ---------------------------------------------------------------------------
DEFAULT_TILE_LIST = Path(
    "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles/parent_tile_list.txt"
)
DEFAULT_TILE_DIR = Path(
    "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles/tiles_100km"
)
DEFAULT_EXPORT_BUCKET = "sidd_rajasthan"
DEFAULT_EXPORT_PREFIX = "psinghal/farmtree_tiles"
DEFAULT_PROJECT = "povertyparking"
DEFAULT_THRESHOLDS = (0.1,)
DEFAULT_MAX_PIXELS = 1e12
DEFAULT_TASK_PREFIX = "farmtree"
DEFAULT_MAX_ACTIVE_TASKS = 3
DEFAULT_POLL_SECONDS = 15

TREE_COLLECTION_IDS = [
    "projects/ee-rscph-2/assets/tree/global",
    "projects/ee-rscph-3/assets/tree/global",
    "projects/ee-rscph-4/assets/tree/global",
    "projects/ee-rscph-5/assets/tree/global",
    "projects/ee-rscph-6/assets/tree/global",
    "projects/ee-rscph-7/assets/tree/global",
    "projects/ee-rscph-8/assets/tree/global",
    "projects/ee-rscph-9/assets/tree/global",
    "projects/ee-rscph-10/assets/tree/global",
    "projects/ee-rscph-11/assets/tree/global",
    "projects/ee-rscph-12/assets/tree/global",
    "projects/ee-rscph-13/assets/tree/global",
    "projects/ee-rscph-14/assets/tree/global",
    "projects/ee-rscph-15/assets/tree/global",
    "projects/ee-rscph-16/assets/tree/global",
    "projects/ee-rscph-17/assets/tree/global",
    "projects/ee-rscph-18/assets/tree/global",
    "projects/ee-rscph-19/assets/tree/global",
    "projects/ee-rscph-20/assets/tree/global",
    "projects/ee-rscph-21/assets/tree/global",
    "projects/ee-rscph-22/assets/tree/global",
    "projects/ee-rscph-23/assets/tree/global",
    "projects/ee-rscph-24/assets/tree/global",
    "projects/ee-rscph-25/assets/tree/global",
    "projects/ee-rscph-26/assets/tree/global",
    "projects/ee-rscph-27/assets/tree/global",
    "projects/ee-rscph-28/assets/tree/global",
    "projects/ee-rscph-29/assets/tree/global",
    "projects/ee-rscph-30/assets/tree/global",
    "projects/ee-rscph-31/assets/tree/global",
    "projects/ee-rscph-47/assets/tree/global",
    "projects/ee-rscph-48/assets/tree/global",
    "projects/ee-rscph-49/assets/tree/global",
]
WORLDCOVER_COLLECTION_ID = "ESA/WorldCover/v200"


# ---------------------------------------------------------------------------
# Data containers & helpers
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class TileMetadata:
    tile_id: str
    path: Path
    width: int
    height: int
    transform: Tuple[float, float, float, float, float, float]
    epsg: int
    bounds: Tuple[float, float, float, float]

    @property
    def crs_transform(self) -> List[float]:
        return list(self.transform)

    @property
    def region(self) -> ee.Geometry:
        xmin, ymin, xmax, ymax = self.bounds
        return ee.Geometry.Rectangle(
            [xmin, ymin, xmax, ymax], proj=f"EPSG:{self.epsg}", geodesic=False
        )


def load_tile_metadata(tile_path: Path) -> TileMetadata:
    if not tile_path.exists():
        raise FileNotFoundError(f"Tile GeoTIFF not found: {tile_path}")

    tile_id = tile_path.stem.replace("tile_", "")
    with rasterio.open(tile_path) as src:  # type: DatasetReader
        width, height = src.width, src.height
        transform = (
            src.transform.a,
            src.transform.b,
            src.transform.c,
            src.transform.d,
            src.transform.e,
            src.transform.f,
        )
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        epsg = src.crs.to_epsg()
        if epsg is None:
            raise ValueError(f"GeoTIFF {tile_path} is missing CRS metadata.")

    return TileMetadata(
        tile_id=tile_id,
        path=tile_path,
        width=width,
        height=height,
        transform=transform,
        epsg=epsg,
        bounds=bounds,
    )


def mosaic_tree_cover() -> ee.Image:
    collections = [ee.ImageCollection(asset) for asset in TREE_COLLECTION_IDS]
    mosaic = collections[0]
    for collection in collections[1:]:
        mosaic = mosaic.merge(collection)
    return mosaic.mosaic()


def load_cropland_mask() -> ee.Image:
    return ee.ImageCollection(WORLDCOVER_COLLECTION_ID).first().eq(40)


def build_mask_image(
    tile_meta: TileMetadata,
    tree_mosaic: ee.Image,
    cropland_mask: ee.Image,
    thresholds: Sequence[float],
) -> ee.Image:
    tile_proj = ee.Projection(f"EPSG:{tile_meta.epsg}")
    transform = tile_meta.crs_transform

    tree_reproj = (
        tree_mosaic.reproject(tile_proj, transform)
        .reduceResolution(reducer=ee.Reducer.mean(), bestEffort=True, maxPixels=1024)
        .reproject(tile_proj, transform)
    )

    cropland_reproj = cropland_mask.reproject(tile_proj, transform)

    mask_bands = []
    for thresh in thresholds:
        band_name = f"tree_gt_{int(thresh * 100)}pct"
        band = tree_reproj.gt(thresh).And(cropland_reproj).unmask(0).rename(band_name)
        mask_bands.append(band)

    return ee.Image(mask_bands).uint8()


def wait_for_task_slots(prefix: str, limit: int, poll_seconds: int) -> None:
    if limit <= 0:
        return

    while True:
        tasks = ee.batch.Task.list()
        active = [
            t
            for t in tasks
            if t.config.get("description", "").startswith(prefix)
            and t.state in ("READY", "RUNNING")
        ]
        if len(active) < limit:
            return
        print(
            f"[controller] {len(active)} tasks in-flight (limit {limit}). Waiting {poll_seconds}s..."
        )
        time.sleep(poll_seconds)


def submit_tile_export(
    tile_meta: TileMetadata,
    mask_image: ee.Image,
    *,
    export_bucket: str,
    export_prefix: str,
    task_prefix: str,
    format_options: Optional[Dict[str, object]],
    max_pixels: float,
    dry_run: bool,
) -> Tuple[Optional[ee.batch.Task], str]:
    description = f"{task_prefix}_{tile_meta.tile_id}"
    file_prefix = f"{export_prefix}/{tile_meta.tile_id}"
    gcs_uri = f"gs://{export_bucket}/{file_prefix}.tif"
    kwargs: Dict[str, object] = {
        "image": mask_image,
        "description": description,
        "bucket": export_bucket,
        "fileNamePrefix": file_prefix,
        "region": tile_meta.region,
        "crs": f"EPSG:{tile_meta.epsg}",
        "crsTransform": tile_meta.crs_transform,
        "maxPixels": max_pixels,
    }
    if format_options:
        kwargs["formatOptions"] = format_options

    print(f"[submit] {tile_meta.tile_id} → {gcs_uri}")
    if dry_run:
        print("         (dry-run: task not started)")
        return None, gcs_uri

    task = ee.batch.Export.image.toCloudStorage(**kwargs)
    task.start()
    print(f"         Task started: {description}")
    return task, gcs_uri


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export farm-tree masks aligned to tessera downloader tiles."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--tile-list-file", type=Path, default=DEFAULT_TILE_LIST)
    parser.add_argument("--tile-dir", type=Path, default=DEFAULT_TILE_DIR)
    parser.add_argument(
        "--tile-paths",
        nargs="+",
        help="Explicit list of tile GeoTIFF paths (overrides tile list file).",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        help="Legacy list of tile IDs to resolve within tile-dir.",
    )
    parser.add_argument("--export-bucket", default=DEFAULT_EXPORT_BUCKET)
    parser.add_argument("--export-prefix", default=DEFAULT_EXPORT_PREFIX)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Tree-cover fractions (0-1) for band thresholds.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Limit the number of tiles processed (useful for quick tests).",
    )
    parser.add_argument(
        "--max-active-tasks",
        type=int,
        default=DEFAULT_MAX_ACTIVE_TASKS,
        help="Maximum concurrent EE tasks tagged with this script.",
    )
    parser.add_argument(
        "--task-prefix",
        default=DEFAULT_TASK_PREFIX,
        help="Prefix for EE task descriptions.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="Seconds to wait between task-queue checks.",
    )
    parser.add_argument(
        "--cloud-optimized",
        action="store_true",
        help="Export Cloud Optimized GeoTIFFs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exports without starting EE tasks.",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        help="Optional path to append JSON status records per submitted tile.",
    )
    return parser.parse_args()


def load_tile_paths(args: argparse.Namespace) -> List[Path]:
    if args.tile_paths:
        paths = [Path(p).expanduser().resolve() for p in args.tile_paths if p.strip()]
    elif args.tiles:
        paths = [
            (args.tile_dir / f"tile_{tid}.tif").resolve()
            for tid in args.tiles
            if tid.strip()
        ]
    else:
        with args.tile_list_file.open() as f:
            ids = [line.strip() for line in f if line.strip()]
        paths = [(args.tile_dir / f"tile_{tid}.tif").resolve() for tid in ids]

    if args.max_tiles is not None:
        paths = paths[: args.max_tiles]
    return paths


def append_status(
    status_path: Optional[Path],
    *,
    tile_id: str,
    description: str,
    task_id: Optional[str],
    gcs_uri: str,
) -> None:
    if not status_path:
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile_id": tile_id,
        "description": description,
        "task_id": task_id,
        "gcs_uri": gcs_uri,
        "timestamp": time.time(),
    }
    with status_path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def schedule_mask_export(
    tile_path: Path,
    *,
    tree_mosaic: ee.Image,
    cropland_mask: ee.Image,
    thresholds: Sequence[float],
    export_bucket: str,
    export_prefix: str,
    task_prefix: str,
    max_pixels: float = DEFAULT_MAX_PIXELS,
    format_options: Optional[Dict[str, object]] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    metadata = load_tile_metadata(tile_path)
    mask_image = build_mask_image(metadata, tree_mosaic, cropland_mask, thresholds)
    task, gcs_uri = submit_tile_export(
        metadata,
        mask_image,
        export_bucket=export_bucket,
        export_prefix=export_prefix,
        task_prefix=task_prefix,
        format_options=format_options,
        max_pixels=max_pixels,
        dry_run=dry_run,
    )
    description = f"{task_prefix}_{metadata.tile_id}"
    return {
        "tile_id": metadata.tile_id,
        "task_id": task.id if task else None,
        "gcs_uri": gcs_uri,
        "description": description,
    }


def main() -> None:
    args = parse_args()

    ee.Initialize(project=args.project)
    print(f"Earth Engine initialized for project '{args.project}'.")

    tile_paths = load_tile_paths(args)
    if not tile_paths:
        print("No tiles to process.")
        return

    print(f"Loaded {len(tile_paths)} tile paths.")

    tree_mosaic = mosaic_tree_cover()
    cropland_mask = load_cropland_mask()

    submitted = 0
    for tile_path in tile_paths:
        try:
            wait_for_task_slots(
                args.task_prefix, args.max_active_tasks, args.poll_seconds
            )
            result = schedule_mask_export(
                tile_path,
                tree_mosaic=tree_mosaic,
                cropland_mask=cropland_mask,
                thresholds=args.thresholds,
                export_bucket=args.export_bucket,
                export_prefix=args.export_prefix,
                task_prefix=args.task_prefix,
                max_pixels=DEFAULT_MAX_PIXELS,
                format_options=(
                    {"cloudOptimized": True} if args.cloud_optimized else None
                ),
                dry_run=args.dry_run,
            )
            append_status(
                args.status_path,
                tile_id=result["tile_id"],
                description=result["description"],
                task_id=result["task_id"],
                gcs_uri=result["gcs_uri"],
            )
            submitted += 1
        except FileNotFoundError as err:
            print(f"[skip] {err}")
            continue

    print(f"Done. Submitted {submitted} tiles (dry-run={args.dry_run}).")


if __name__ == "__main__":
    main()
