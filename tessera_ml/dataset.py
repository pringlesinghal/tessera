import os
import bisect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq


class TreeDataset(Dataset):
    """
    Dataset for loading tree pixel timeseries from globally shuffled Parquet shards.

    Structure:
    - Index: Directory of globally sorted/shuffled part-XXXX.parquet files.
    - Data:  Numpy files (bands.npy, etc.) stored in {data_dir}/{year}/{tile_id}/data_processed/

    Strategy: "Chunked Reading"
    - Instead of loading the full 1.25B row index, we scan the file lengths.
    - We map a global index `idx` to (file_index, local_index).
    - We keep the *active* dataframe in memory.
    - Since data is globally shuffled, sequential access (0->N) yields random samples.
    """

    def __init__(self, index_dir, data_dir, year=None, years=None, cache_size=50):
        """
        Args:
            year (int): Fixed year to load (e.g. 2023). If None, will pick randomly from 'years'.
            years (list): List of available years to sample from (e.g. [2016, 2017...]).
                          Required if year is None.
        """
        self.index_dir = Path(index_dir)
        self.data_dir = Path(data_dir)
        self.year = year
        self.years = years if years else list(range(2016, 2025))
        self.cache_size = cache_size

        # 1. Scan Index Files
        print(f"Scanning index files in {self.index_dir}...")
        self.index_files = sorted(list(self.index_dir.glob("*.parquet")))
        if not self.index_files:
            raise FileNotFoundError(f"No .parquet files found in {self.index_dir}")

        # 2. Build Cumulative Index (Lightweight)
        # We need to know how many rows are in each file to support random access __getitem__
        self.file_offsets = [0]
        total_rows = 0

        for f in self.index_files:
            # Use PyArrow to read metadata only (fast)
            meta = pq.read_metadata(f)
            rows = meta.num_rows
            total_rows += rows
            self.file_offsets.append(total_rows)

        self.total_length = total_rows
        print(
            f"Index scanned. Total samples: {self.total_length:,} across {len(self.index_files)} files."
        )

        # 3. State for Index Caching
        self.current_file_idx = -1
        self.current_df = None

        # 4. Cache for Data Tiles
        self.tile_cache = {}
        self.cache_queue = []

    def _load_index_chunk(self, file_idx):
        """Loads a specific parquet file into memory."""
        if file_idx == self.current_file_idx:
            return

        # Load new file
        path = self.index_files[file_idx]
        # print(f"[DEBUG] Loading index chunk: {path.name}")
        self.current_df = pd.read_parquet(path)
        self.current_file_idx = file_idx

    def _get_tile_data(self, tile_id, year):
        # Create a unique cache key combining tile_id and year
        cache_key = f"{tile_id}_{year}"

        # LRU Cache logic (Same as before)
        if cache_key in self.tile_cache:
            if cache_key in self.cache_queue:
                self.cache_queue.remove(cache_key)
            self.cache_queue.append(cache_key)
            return self.tile_cache[cache_key]

        if len(self.cache_queue) >= self.cache_size:
            oldest = self.cache_queue.pop(0)
            if oldest in self.tile_cache:
                del self.tile_cache[oldest]

        # Load new tile
        tile_path = self.data_dir / str(year) / tile_id / "data_processed"

        if not tile_path.exists():
            raise FileNotFoundError(f"Tile data not found: {tile_path}")

        try:
            arrays = {}
            arrays["bands"] = np.load(tile_path / "bands.npy", mmap_mode="r")
            arrays["masks"] = np.load(tile_path / "masks.npy", mmap_mode="r")
            arrays["doys"] = np.load(tile_path / "doys.npy", mmap_mode="r")

            if (tile_path / "sar_ascending.npy").exists():
                arrays["sar_asc"] = np.load(
                    tile_path / "sar_ascending.npy", mmap_mode="r"
                )
                arrays["sar_asc_doy"] = np.load(
                    tile_path / "sar_ascending_doy.npy", mmap_mode="r"
                )
            else:
                arrays["sar_asc"] = None

            if (tile_path / "sar_descending.npy").exists():
                arrays["sar_desc"] = np.load(
                    tile_path / "sar_descending.npy", mmap_mode="r"
                )
                arrays["sar_desc_doy"] = np.load(
                    tile_path / "sar_descending_doy.npy", mmap_mode="r"
                )
            else:
                arrays["sar_desc"] = None

            self.tile_cache[cache_key] = arrays
            self.cache_queue.append(cache_key)
            return arrays

        except Exception as e:
            raise RuntimeError(f"Failed to load tile {tile_id}: {e}")

    def __len__(self):
        return self.total_length

    def get_sample(self, idx, year=None):
        """Explicitly load a sample for a specific year."""
        return self.__getitem__(idx, year_override=year)

    def __getitem__(self, global_idx, year_override=None):
        # Determine year: override > self.year > random
        if year_override is not None:
            target_year = year_override
        elif self.year is not None:
            target_year = self.year
        else:
            target_year = int(np.random.choice(self.years))

        # 1. Map global_idx to file_idx and local_idx
        # bisect_right returns insertion point.
        # offsets: [0, 1000, 2000]
        # idx 500 -> bisect_right gives 1. file_idx = 1-1 = 0.
        file_idx = bisect.bisect_right(self.file_offsets, global_idx) - 1
        local_idx = global_idx - self.file_offsets[file_idx]

        # 2. Ensure correct index chunk is loaded
        self._load_index_chunk(file_idx)

        # 3. Get metadata
        record = self.current_df.iloc[local_idx]
        tile_id = record["tile_id"]
        row = record["row"]
        col = record["col"]

        # 4. Get Data
        try:
            tile_data = self._get_tile_data(tile_id, target_year)
        except (FileNotFoundError, RuntimeError) as e:
            # print(f"[WARN] Sample {global_idx} failed: {e}")
            raise e

        # 5. Slice & Return (Same as before)
        bands_data = np.array(tile_data["bands"][:, row, col, :])
        masks_data = np.array(tile_data["masks"][:, row, col])
        doys_data = np.array(tile_data["doys"][:])

        if tile_data.get("sar_asc") is not None and tile_data["sar_asc"].shape[0] > 0:
            sar_asc_data = torch.from_numpy(
                np.array(tile_data["sar_asc"][:, row, col, :])
            ).float()
            sar_asc_doy = torch.from_numpy(np.array(tile_data["sar_asc_doy"][:])).long()
        else:
            sar_asc_data = None
            sar_asc_doy = None

        if tile_data.get("sar_desc") is not None and tile_data["sar_desc"].shape[0] > 0:
            sar_desc_data = torch.from_numpy(
                np.array(tile_data["sar_desc"][:, row, col, :])
            ).float()
            sar_desc_doy = torch.from_numpy(
                np.array(tile_data["sar_desc_doy"][:])
            ).long()
        else:
            sar_desc_data = None
            sar_desc_doy = None

        return {
            "tile_id": tile_id,
            "coords": torch.tensor([row, col], dtype=torch.long),
            "year": target_year,
            "bands": torch.from_numpy(bands_data).float(),
            "masks": torch.from_numpy(masks_data).long(),
            "doys": torch.from_numpy(doys_data).long(),
            "sar_asc": sar_asc_data,
            "sar_asc_doy": sar_asc_doy,
            "sar_desc": sar_desc_data,
            "sar_desc_doy": sar_desc_doy,
        }


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    print("\n--- Initializing Chunked Dataset (Random Year Mode) ---")
    # Initialize with year=None to test random selection
    ds = TreeDataset(args.index_dir, args.data_dir, year=None)

    if len(ds) > 0:
        print("\n--- Loading First Sample (Index 0) ---")
        idx = 0

        # Load twice to demonstrate random year variation
        for i in range(2):
            print(f"\n[Attempt {i+1}] Loading index {idx}...")
            start_t = time.time()
            sample = ds[idx]
            dur = time.time() - start_t

            print(f"Loaded sample {idx} in {dur:.4f} sec")
            print(f"Tile: {sample['tile_id']}")
            print(f"Year: {sample['year']}")
            print(f"Coords: {sample['coords']}")
            print(f"Bands Shape: {sample['bands'].shape}")

            # Basic data check
            if sample["bands"].shape[0] > 0:
                print(f"First band mean: {sample['bands'][0].mean():.2f}")
            else:
                print("Warning: Bands are empty/missing")

    else:
        print("Dataset is empty.")
