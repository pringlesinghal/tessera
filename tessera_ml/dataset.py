import os
# tessera_ml/dataset.py

import bisect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Normalization parameters
S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                         2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                        1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)

S1_BAND_MEAN = np.array([5484.0407, 3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334, 1726.0670], dtype=np.float32)


class TreeDataset(Dataset):
    """
    Dataset for multimodal (S2 + S1) time series training.
    
    Structure:
    - Index: Directory of globally sorted/shuffled part-XXXX.parquet files.
    - Data:  Numpy files (bands.npy, etc.) stored in {data_dir}/{year}/{tile_id}/data_processed/

    Strategy: "Chunked Reading"
    - Scans index files to build a global map.
    - Loads data on demand using LRU cache.
    - Performs online temporal sampling to generate 'aug1' and 'aug2' views.
    """

    def __init__(self, index_dir, data_dir, year=None, years=None, cache_size=50,
                 sample_size_s2=20, sample_size_s1=20, normalize=True):
        """
        Args:
            index_dir: Directory containing parquet index files
            data_dir: Root directory containing year subdirectories with tile data
            year: If specified, only use this year. Otherwise random.
            years: If specified, list of years to sample from. Otherwise all available.
            cache_size: Number of tiles to keep in memory
            sample_size_s2: Number of timesteps to sample for S2
            sample_size_s1: Number of timesteps to sample for S1
            normalize: Whether to normalize the data
        """
        
        # Setup corrupted tiles log file
        self.corrupted_log_path = Path(data_dir) / "corrupted_tiles.txt"
        self.index_dir = Path(index_dir)
        self.data_dir = Path(data_dir)
        self.year = year
        self.years = years if years else list(range(2016, 2025))
        self.cache_size = cache_size
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.normalize = normalize

        # 1. Scan Index Files
        print(f"Scanning index files in {self.index_dir}...")
        self.index_files = sorted(list(self.index_dir.glob("part.*.parquet")))
        if not self.index_files:
            raise FileNotFoundError(f"No .parquet files found in {self.index_dir}")

        # 2. Build Cumulative Index (Lightweight)
        self.file_offsets = [0]
        total_rows = 0

        for f in self.index_files:
            meta = pq.read_metadata(f)
            rows = meta.num_rows
            total_rows += rows
            self.file_offsets.append(total_rows)

        self.total_length = total_rows
        print(f"Found {self.total_length} samples across {len(self.index_files)} files.")

        # 3. Cache State
        self.current_file_idx = -1
        self.current_df = None
        
        # Tile Data Cache: { "tile_id_year": { "bands": memmap, ... } }
        self.tile_cache = {}
        self.cache_queue = [] # For LRU eviction

    def _load_index_chunk(self, file_idx):
        """Loads a specific parquet index file into memory."""
        if file_idx == self.current_file_idx:
            return

        t0 = time.time()
        path = self.index_files[file_idx]
        self.current_df = pd.read_parquet(path)
        self.current_file_idx = file_idx
        logger.debug(f"Loaded index chunk {path.name} in {time.time()-t0:.3f}s")

    def _get_tile_data(self, tile_id, year):
        """
        Retrieves tile data arrays from cache or disk.
        Returns dictionary of memmap arrays.
        """
        t0 = time.time()
        cache_key = f"{tile_id}_{year}"

        if cache_key in self.tile_cache:
            if cache_key in self.cache_queue:
                self.cache_queue.remove(cache_key)
            self.cache_queue.append(cache_key)
            logger.debug(f"Cache hit for {cache_key} in {time.time()-t0:.4f}s")
            return self.tile_cache[cache_key]

        # Evict if full
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
            # S2 Data (Required)
            arrays["bands"] = np.load(tile_path / "bands.npy", mmap_mode="r")
            arrays["masks"] = np.load(tile_path / "masks.npy", mmap_mode="r")
            arrays["doys"] = np.load(tile_path / "doys.npy", mmap_mode="r")
            
            # S1 Data (Optional/Check existence)
            if (tile_path / "sar_ascending.npy").exists():
                arrays["sar_asc"] = np.load(tile_path / "sar_ascending.npy", mmap_mode="r")
                arrays["sar_asc_doy"] = np.load(tile_path / "sar_ascending_doy.npy", mmap_mode="r")
            else:
                arrays["sar_asc"] = None

            if (tile_path / "sar_descending.npy").exists():
                arrays["sar_desc"] = np.load(tile_path / "sar_descending.npy", mmap_mode="r")
                arrays["sar_desc_doy"] = np.load(tile_path / "sar_descending_doy.npy", mmap_mode="r")
            else:
                arrays["sar_desc"] = None

            self.tile_cache[cache_key] = arrays
            self.cache_queue.append(cache_key)
            logger.debug(f"Loaded tile {cache_key} from disk in {time.time()-t0:.3f}s")
            return arrays

        except Exception as e:
            raise RuntimeError(f"Failed to load tile {tile_id}: {e}")

    def _sample_indices(self, valid_indices, size):
        """Samples `size` indices from `valid_indices` with replacement if needed."""
        if len(valid_indices) == 0:
            return np.array([], dtype=int)
        
        if len(valid_indices) < size:
            # Resample with replacement to fill size
            return np.random.choice(valid_indices, size, replace=True)
        else:
            # Sample without replacement
            return np.random.choice(valid_indices, size, replace=False)

    def _process_s2_sample(self, bands, masks, doys, indices):
        """
        Extracts S2 data for given indices, normalizes, and appends DOY.
        Output: (T, 11)
        """
        if len(indices) == 0:
            return np.zeros((self.sample_size_s2, 11), dtype=np.float32)

        # Sort indices to preserve temporal order (optional but good practice)
        indices = np.sort(indices)
        
        selected_bands = bands[indices].astype(np.float32) # (T, 10)
        selected_doys = doys[indices].astype(np.float32)   # (T,)

        if self.normalize:
            selected_bands = (selected_bands - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)

        # Append DOY
        return np.hstack([selected_bands, selected_doys.reshape(-1, 1)])

    def _process_s1_sample(self, asc_bands, asc_doys, desc_bands, desc_doys, size):
        """
        Combines Asc/Desc S1 data, samples `size` steps, normalizes, and appends DOY.
        Output: (T, 3)
        """
        # 1. Collect all valid S1 observations
        # We assume input bands are already sliced for the pixel [:, row, col, :]
        # But we need to check validity (not all zeros)
        
        valid_obs = []
        
        # Helper to process one orbit direction
        def collect_valid(bands, doys):
            if bands is None: return
            # Check for non-zero data (assuming 0 is missing/padding)
            # Or just take all if we trust the file structure. 
            # Usually S1 data is dense in the file but might be missing for some dates.
            # Let's assume all entries in the .npy are valid acquisitions for that tile,
            # but we need to check if the specific pixel has data (not nodata).
            # A simple check is if sum(abs(bands)) > 0
            
            # Vectorized check for valid pixels in the time series
            is_valid = np.any(bands != 0, axis=1) # (T,)
            valid_idx = np.where(is_valid)[0]
            
            for idx in valid_idx:
                valid_obs.append((bands[idx], doys[idx]))

        collect_valid(asc_bands, asc_doys)
        collect_valid(desc_bands, desc_doys)

        if not valid_obs:
            return np.zeros((size, 3), dtype=np.float32)

        # 2. Sample
        # We have a list of (band_data, doy).
        # We need to pick `size` random ones.
        indices = np.random.choice(len(valid_obs), size, replace=(len(valid_obs) < size))
        indices = np.sort(indices) # Sort by index in the list (not necessarily time)
        
        # To sort by time, we should extract DOYs first
        sampled_obs = [valid_obs[i] for i in indices]
        
        # Sort by DOY
        sampled_obs.sort(key=lambda x: x[1])
        
        # 3. Construct Tensor
        out_bands = np.array([x[0] for x in sampled_obs], dtype=np.float32)
        out_doys = np.array([x[1] for x in sampled_obs], dtype=np.float32)
        
        if self.normalize:
            out_bands = (out_bands - S1_BAND_MEAN) / (S1_BAND_STD + 1e-9)
            
        return np.hstack([out_bands, out_doys.reshape(-1, 1)])

    def __len__(self):
        return self.total_length

    def get_sample(self, idx, year=None):
        """Explicitly load a sample for a specific year."""
        return self.__getitem__(idx, year_override=year)

    def __getitem__(self, global_idx, year_override=None):
        t_total = time.time()
        # Determine year
        if year_override is not None:
            target_year = year_override
        elif self.year is not None:
            target_year = self.year
        else:
            target_year = int(np.random.choice(self.years))

        # 1. Map global_idx to file_idx and local_idx
        file_idx = bisect.bisect_right(self.file_offsets, global_idx) - 1
        local_idx = global_idx - self.file_offsets[file_idx]

        # 2. Load Index Chunk
        self._load_index_chunk(file_idx)

        # 3. Get Tile Info
        row_data = self.current_df.iloc[local_idx]
        tile_id = row_data["tile_id"]
        row = row_data["row"]
        col = row_data["col"]

        # Load tile data
        try:
            tile_data = self._get_tile_data(tile_id, target_year)
        except (ValueError, RuntimeError, OSError) as e:
            # Skip corrupted or missing tiles
            logger.warning(f"Skipping corrupted tile {tile_id} (year {target_year}): {e}")
            
            # Log to file for later investigation
            try:
                with open(self.corrupted_log_path, 'a') as f:
                    f.write(f"{tile_id},{target_year}\n")
            except Exception as log_err:
                logger.error(f"Failed to log corrupted tile: {log_err}")
            
            # Return a random different sample instead
            new_idx = (global_idx + 1) % len(self)
            return self.__getitem__(new_idx, year_override=year_override)
        except Exception as e:
            logger.error(f"Unexpected error loading tile {tile_id}: {e}")
            raise e

        # 5. S2 Processing
        bands_s2 = tile_data["bands"][:, row, col, :]
        masks_s2 = tile_data["masks"][:, row, col]
        doys_s2 = tile_data["doys"][:]
        
        # Identify valid S2 indices (mask == 1 is valid/cloud-free? 
        # In previous code: mask==1 -> valid (2), mask==0 -> cloud (1).
        # Let's check `_process_time_series` in previous version:
        # if masks[t] == 1: new_mask[pos] = 2 (Valid)
        # else: new_mask[pos] = 1 (Cloud)
        # So mask==1 is the good data.
        valid_s2_indices = np.where(masks_s2 == 1)[0]
        
        # Generate two views for S2
        idx_s2_1 = self._sample_indices(valid_s2_indices, self.sample_size_s2)
        idx_s2_2 = self._sample_indices(valid_s2_indices, self.sample_size_s2)
        
        s2_aug1 = self._process_s2_sample(bands_s2, masks_s2, doys_s2, idx_s2_1)
        s2_aug2 = self._process_s2_sample(bands_s2, masks_s2, doys_s2, idx_s2_2)

        # 6. S1 Processing
        # Extract S1 data if available
        sar_asc = tile_data["sar_asc"][:, row, col, :] if tile_data["sar_asc"] is not None else None
        sar_asc_doy = tile_data["sar_asc_doy"][:] if tile_data["sar_asc"] is not None else None
        
        sar_desc = tile_data["sar_desc"][:, row, col, :] if tile_data["sar_desc"] is not None else None
        sar_desc_doy = tile_data["sar_desc_doy"][:] if tile_data["sar_desc"] is not None else None
        
        # Generate two views for S1
        # Note: _process_s1_sample handles sampling internally because it combines asc/desc
        s1_aug1 = self._process_s1_sample(sar_asc, sar_asc_doy, sar_desc, sar_desc_doy, self.sample_size_s1)
        s1_aug2 = self._process_s1_sample(sar_asc, sar_asc_doy, sar_desc, sar_desc_doy, self.sample_size_s1)

        result = {
            "s2_aug1": torch.from_numpy(s2_aug1),
            "s2_aug2": torch.from_numpy(s2_aug2),
            "s1_aug1": torch.from_numpy(s1_aug1),
            "s1_aug2": torch.from_numpy(s1_aug2)
        }
        
        t_elapsed = time.time() - t_total
        if global_idx % 1000 == 0:  # Log every 1000 samples
            logger.info(f"Sample {global_idx} loaded in {t_elapsed:.4f}s")
        
        return result


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    print("\n--- Initializing TreeDataset (Multimodal) ---")
    ds = TreeDataset(args.index_dir, args.data_dir, year=None)

    if len(ds) > 0:
        print("\n--- Loading First Sample (Index 0) ---")
        idx = 0
        start_t = time.time()
        sample = ds[idx]
        dur = time.time() - start_t

        print(f"Loaded sample {idx} in {dur:.4f} sec")
        for k, v in sample.items():
            print(f"{k}: {v.shape}, dtype={v.dtype}")
            if v.shape[0] > 0:
                print(f"  Mean: {v.mean():.2f}")

    else:
        print("Dataset is empty.")
