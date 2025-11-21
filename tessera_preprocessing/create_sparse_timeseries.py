#!/usr/bin/env python3
"""
Given a processed tile folder (with bands.npy, masks.npy, etc.) and the corresponding
tree-mask GeoTIFF, create sparse tensors:
 - optical_timeseries.npy  (T x P x B)
 - optical_valid.npy       (P x T)
 - optical_idx.npy         (P x 2) -> (row, col)
Similar outputs are generated for SAR ascending/descending when data exists.
"""

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio


def load_tree_mask(mask_tif: Path, band_index: int) -> np.ndarray:
    with rasterio.open(mask_tif) as src:
        if band_index < 0 or band_index >= src.count:
            raise ValueError(
                f"Mask band_index { band_index } out of range (0-{src.count - 1})"
            )
        mask = src.read(band_index + 1)
    return np.asarray(mask > 0, dtype=bool)


def parse_subtile_indices(tile_id: str) -> Tuple[int, int]:
    """
    Expect tile IDs like 'zoneXXXX_rYYY_cZZZ_srA_scB'. Returns (sr, sc).
    """
    match = re.search(r"_sr(\d+)_sc(\d+)", tile_id)
    if not match:
        raise ValueError(
            f"Cannot determine sub-tile indices from tile id '{tile_id}'. "
            "Expected suffix '_sr<row>_sc<col>'."
        )
    return int(match.group(1)), int(match.group(2))


def gather_cube(
    cube: np.memmap, rows: np.ndarray, cols: np.ndarray, desc: str
) -> np.ndarray:
    if cube.size == 0:
        return np.empty((0, rows.size, 0))
    T = cube.shape[0]
    B = cube.shape[3]
    result = np.empty((T, rows.size, B), dtype=cube.dtype)
    for t in range(T):
        result[t] = cube[t][rows, cols, :]
        if t % 32 == 0:
            print(f"[{desc}] processed slice {t+1}/{T}")
    return result


def gather_masks(
    mask_cube: np.memmap, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    T = mask_cube.shape[0]
    result = np.empty((rows.size, T), dtype=bool)
    for t in range(T):
        result[:, t] = mask_cube[t][rows, cols]
    return result


def main(processed_dir: Path, mask_tif: Path, band_index: int) -> None:
    processed_dir = processed_dir.resolve()
    mask_tif = mask_tif.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(processed_dir)
    if not mask_tif.exists():
        raise FileNotFoundError(mask_tif)

    print(f"Processed dir: {processed_dir}")
    print(f"Tree mask: {mask_tif}")

    bands = np.load(processed_dir / "bands.npy", mmap_mode="r")
    masks = np.load(processed_dir / "masks.npy", mmap_mode="r").astype(bool)
    tree_mask = load_tree_mask(mask_tif, band_index)

    tile_h, tile_w = masks.shape[1:]
    mask_h, mask_w = tree_mask.shape
    if (mask_h, mask_w) != (tile_h, tile_w):
        if mask_h % tile_h != 0 or mask_w % tile_w != 0:
            raise ValueError(
                f"Tree mask shape {tree_mask.shape} is not a multiple of tile shape {(tile_h, tile_w)}"
            )
        tiles_y = mask_h // tile_h
        tiles_x = mask_w // tile_w
        tile_id = processed_dir.parent.name
        sr, sc = parse_subtile_indices(tile_id)
        if sr >= tiles_y or sc >= tiles_x:
            raise ValueError(
                f"Subtile indices (sr={sr}, sc={sc}) exceed parent grid ({tiles_y}, {tiles_x})"
            )
        y0 = sr * tile_h
        y1 = y0 + tile_h
        x0 = sc * tile_w
        x1 = x0 + tile_w
        tree_mask = tree_mask[y0:y1, x0:x1]

    mask_any = masks.any(axis=0)
    pixel_mask = tree_mask & mask_any
    rows, cols = np.where(pixel_mask)
    if rows.size == 0:
        print("No valid pixels found after masking.")
        return

    optical_ts = gather_cube(bands, rows, cols, "optical")
    optical_valid = gather_masks(masks, rows, cols)

    sar_asc = np.load(processed_dir / "sar_ascending.npy", mmap_mode="r")
    sar_desc = np.load(processed_dir / "sar_descending.npy", mmap_mode="r")

    np.save(processed_dir / "optical_timeseries.npy", optical_ts)
    np.save(processed_dir / "optical_valid.npy", optical_valid)
    np.save(processed_dir / "optical_idx.npy", np.stack([rows, cols], axis=1))

    tree_valid = tree_mask[rows, cols]

    if sar_asc.size > 0:
        sar_asc_ts = gather_cube(sar_asc, rows, cols, "sar_asc")
        sar_asc_valid = np.broadcast_to(
            tree_valid[:, None], (tree_valid.size, sar_asc.shape[0])
        )
        np.save(processed_dir / "sar_asc_timeseries.npy", sar_asc_ts)
        np.save(processed_dir / "sar_asc_valid.npy", sar_asc_valid)

    if sar_desc.size > 0:
        sar_desc_ts = gather_cube(sar_desc, rows, cols, "sar_desc")
        sar_desc_valid = np.broadcast_to(
            tree_valid[:, None], (tree_valid.size, sar_desc.shape[0])
        )
        np.save(processed_dir / "sar_desc_timeseries.npy", sar_desc_ts)
        np.save(processed_dir / "sar_desc_valid.npy", sar_desc_valid)

    print(f"Saved sparse arrays in {processed_dir}")
    print(f"  Pixels kept: {rows.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sparse timeseries tensors.")
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("tree_mask_tif", type=Path)
    parser.add_argument(
        "--mask-band-index",
        type=int,
        default=0,
        help="0-based band index to use from the tree mask TIFF (default: 0).",
    )
    args = parser.parse_args()
    main(args.processed_dir, args.tree_mask_tif, args.mask_band_index)
