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
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio


def load_tree_mask(mask_tif: Path) -> np.ndarray:
    with rasterio.open(mask_tif) as src:
        mask = src.read(1)
    return mask > 0


def flatten_indices(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mask: bool array of shape (T, H, W)
    Returns (valid_idx (P,), rows (P,), cols (P,))
    """
    T, H, W = mask.shape
    flat = mask.reshape(T, H * W)
    valid_columns = flat.any(axis=0)
    valid_idx = np.where(valid_columns)[0]
    rows = valid_idx // W
    cols = valid_idx % W
    return valid_idx, rows, cols


def filter_cube(cube: np.ndarray, valid_idx: np.ndarray) -> Optional[np.ndarray]:
    if cube.size == 0:
        return None
    T, H, W, B = cube.shape
    reshaped = cube.reshape(T, H * W, B)
    return reshaped[:, valid_idx, :]


def filter_sar(cube: np.ndarray, valid_idx: np.ndarray) -> Optional[np.ndarray]:
    if cube.size == 0:
        return None
    T, H, W, B = cube.shape
    reshaped = cube.reshape(T, H * W, B)
    return reshaped[:, valid_idx, :]


def save_outputs(
    out_dir: Path,
    prefix: str,
    data: Optional[np.ndarray],
    valid_mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    if data is None:
        return
    np.save(out_dir / f"{prefix}_timeseries.npy", data)
    np.save(out_dir / f"{prefix}_valid.npy", valid_mask)
    np.save(out_dir / f"{prefix}_idx.npy", np.stack([rows, cols], axis=1))


def main(processed_dir: Path, mask_tif: Path) -> None:
    processed_dir = processed_dir.resolve()
    mask_tif = mask_tif.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(processed_dir)
    if not mask_tif.exists():
        raise FileNotFoundError(mask_tif)

    print(f"Processed dir: {processed_dir}")
    print(f"Tree mask: {mask_tif}")

    bands = np.load(processed_dir / "bands.npy")
    masks = np.load(processed_dir / "masks.npy").astype(bool)
    tree_mask = load_tree_mask(mask_tif)

    if tree_mask.shape != masks.shape[1:]:
        raise ValueError(
            f"Tree mask shape {tree_mask.shape} does not match data shape {masks.shape[1:]}."
        )

    combined_mask = masks & tree_mask[None, ...]
    valid_idx, rows, cols = flatten_indices(combined_mask)
    if valid_idx.size == 0:
        print("No valid pixels found after masking.")
        return

    optical_ts = filter_cube(bands, valid_idx)
    optical_valid = combined_mask.reshape(combined_mask.shape[0], -1)[:, valid_idx].T

    sar_asc = np.load(processed_dir / "sar_ascending.npy")
    sar_desc = np.load(processed_dir / "sar_descending.npy")

    sar_asc_valid = (
        np.broadcast_to(tree_mask.reshape(1, -1), (sar_asc.shape[0], tree_mask.size))[:, valid_idx].T
        if sar_asc.size > 0
        else None
    )
    sar_desc_valid = (
        np.broadcast_to(tree_mask.reshape(1, -1), (sar_desc.shape[0], tree_mask.size))[:, valid_idx].T
        if sar_desc.size > 0
        else None
    )

    sar_asc_ts = filter_sar(sar_asc, valid_idx)
    sar_desc_ts = filter_sar(sar_desc, valid_idx)

    save_outputs(processed_dir, "optical", optical_ts, optical_valid, rows, cols)
    if sar_asc_ts is not None:
        save_outputs(processed_dir, "sar_asc", sar_asc_ts, sar_asc_valid, rows, cols)
    if sar_desc_ts is not None:
        save_outputs(processed_dir, "sar_desc", sar_desc_ts, sar_desc_valid, rows, cols)

    print(f"Saved sparse arrays in {processed_dir}")
    print(f"  Pixels kept: {len(valid_idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sparse timeseries tensors.")
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("tree_mask_tif", type=Path)
    args = parser.parse_args()
    main(args.processed_dir, args.tree_mask_tif)

