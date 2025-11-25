import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq

class TreeDataset(Dataset):
    """
    Dataset for loading tree pixel timeseries.
    
    Structure:
    - Index: Parquet files containing (tile_id, row, col) for every tree.
    - Data:  Numpy files (bands.npy, etc.) stored in {data_dir}/{year}/{tile_id}/data_processed/
    
    This dataset performs "Lazy Loading":
    1. It keeps the lightweight index in memory.
    2. When __getitem__ is called, it identifies the tile and pixel coordinates.
    3. It uses a Least-Recently-Used (LRU) cache (or simple dict for now) to keep
       open file handles to the large .npy files, using mmap_mode='r'.
    4. It slices the specific pixel's time series from disk.
    """
    def __init__(self, index_dir, data_dir, year=2023, cache_size=50):
        """
        Args:
            index_dir (str): Path to directory containing part-XXXXX.parquet index files.
            data_dir (str): Root path to time series data (e.g. /scratch/.../time_series)
            year (int): Year to load data for.
            cache_size (int): Max number of open tile file handles to keep in memory.
        """
        self.index_dir = Path(index_dir)
        self.data_dir = Path(data_dir)
        self.year = year
        self.cache_size = cache_size
        
        # Load Index
        print(f"Loading index from {self.index_dir}...")
        self.index_df = self._load_index()
        print(f"Index loaded. Total samples: {len(self.index_df):,}")
        
        # Cache for open numpy memory maps: { 'tile_id': { 'bands': memmap, ... } }
        self.tile_cache = {}
        self.cache_queue = [] # To track LRU order

    def _load_index(self):
        """Reads all parquet shards into a single DataFrame."""
        # Check if directory exists
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")
            
        files = sorted(list(self.index_dir.glob("*.parquet")))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {self.index_dir}")
            
        # Read all files
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def _get_tile_data(self, tile_id):
        """
        Retrieves memory-mapped arrays for a tile, managing the cache.
        """
        # 1. Return if already in cache
        if tile_id in self.tile_cache:
            # Move to end of queue (mark as recently used)
            if tile_id in self.cache_queue:
                self.cache_queue.remove(tile_id)
            self.cache_queue.append(tile_id)
            return self.tile_cache[tile_id]

        # 2. Evict oldest if cache is full
        if len(self.cache_queue) >= self.cache_size:
            oldest_tile = self.cache_queue.pop(0)
            if oldest_tile in self.tile_cache:
                # Closing mmap is tricky in Python, usually we just delete the reference
                # and let GC handle it, but explicit closing isn't supported for memmap.
                del self.tile_cache[oldest_tile]

        # 3. Load new tile
        # Path: {data_dir}/{year}/{tile_id}/data_processed/
        tile_path = self.data_dir / str(self.year) / tile_id / "data_processed"
        
        if not tile_path.exists():
             # Fallback or error - for now raise error to be visible
             raise FileNotFoundError(f"Tile data not found: {tile_path}")

        # Load specific files we need
        # Using mmap_mode='r' is CRITICAL for speed and memory
        try:
            arrays = {}
            arrays['bands'] = np.load(tile_path / "bands.npy", mmap_mode='r')
            arrays['masks'] = np.load(tile_path / "masks.npy", mmap_mode='r')
            arrays['doys']  = np.load(tile_path / "doys.npy", mmap_mode='r')
            
            # Optional SAR files (might be empty with shape T=0)
            if (tile_path / "sar_ascending.npy").exists():
                 arrays['sar_asc'] = np.load(tile_path / "sar_ascending.npy", mmap_mode='r')
                 arrays['sar_asc_doy'] = np.load(tile_path / "sar_ascending_doy.npy", mmap_mode='r')
            else:
                 arrays['sar_asc'] = None

            if (tile_path / "sar_descending.npy").exists():
                 arrays['sar_desc'] = np.load(tile_path / "sar_descending.npy", mmap_mode='r')
                 arrays['sar_desc_doy'] = np.load(tile_path / "sar_descending_doy.npy", mmap_mode='r')
            else:
                 arrays['sar_desc'] = None

            # Add to cache
            self.tile_cache[tile_id] = arrays
            self.cache_queue.append(tile_id)
            return arrays
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tile {tile_id}: {e}")

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        # 1. Get metadata from index
        record = self.index_df.iloc[idx]
        tile_id = record['tile_id']
        row = record['row']
        col = record['col']
        
        # 2. Get dense arrays for this tile (cached)
        try:
            tile_data = self._get_tile_data(tile_id)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[WARN] Sample {idx} failed: {e}")
            raise e

        # 3. Slice the specific pixel
        
        # Optical
        bands_data = np.array(tile_data['bands'][:, row, col, :]) # Shape: (T_opt, B_opt)
        masks_data = np.array(tile_data['masks'][:, row, col])    # Shape: (T_opt,)
        doys_data  = np.array(tile_data['doys'][:])               # Shape: (T_opt,)
        
        # SAR Ascending
        if tile_data.get('sar_asc') is not None and tile_data['sar_asc'].shape[0] > 0:
            sar_asc_data = torch.from_numpy(np.array(tile_data['sar_asc'][:, row, col, :])).float() # Shape: (T_sar, 2)
            sar_asc_doy  = torch.from_numpy(np.array(tile_data['sar_asc_doy'][:])).long()          # Shape: (T_sar,)
        else:
            sar_asc_data = None
            sar_asc_doy  = None

        # SAR Descending
        if tile_data.get('sar_desc') is not None and tile_data['sar_desc'].shape[0] > 0:
            sar_desc_data = torch.from_numpy(np.array(tile_data['sar_desc'][:, row, col, :])).float() # Shape: (T_sar, 2)
            sar_desc_doy  = torch.from_numpy(np.array(tile_data['sar_desc_doy'][:])).long()          # Shape: (T_sar,)
        else:
            sar_desc_data = None
            sar_desc_doy  = None
        
        # 4. Convert to Torch Tensors
        return {
            'tile_id': tile_id,
            'coords': torch.tensor([row, col], dtype=torch.long),
            'year': self.year,
            'bands': torch.from_numpy(bands_data).float(),
            'masks': torch.from_numpy(masks_data).long(),
            'doys': torch.from_numpy(doys_data).long(),
            'sar_asc': sar_asc_data,
            'sar_asc_doy': sar_asc_doy,
            'sar_desc': sar_desc_data,
            'sar_desc_doy': sar_desc_doy,
        }

if __name__ == "__main__":
    # Test Block
    import argparse
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    
    print("\n--- Initializing Dataset ---")
    ds = TreeDataset(args.index_dir, args.data_dir, year=2016)
    
    if len(ds) > 0:
        print("\n--- Loading Random Sample ---")
        idx = np.random.randint(0, len(ds))
        
        start_t = time.time()
        sample = ds[idx]
        dur = time.time() - start_t
        
        print(f"Loaded sample {idx} in {dur:.4f} sec")
        print(f"Tile: {sample['tile_id']}")
        print(f"Coords: {sample['coords']}")
        print(f"Bands Shape: {sample['bands'].shape} (Expected T x B)")
        print(f"Masks Shape: {sample['masks'].shape} (Expected T)")
        print(f"DOYs Shape:  {sample['doys'].shape}  (Expected T)")
        print(f"SAR Asc Shape: {sample['sar_asc'].shape} (Expected T_sar x 2)")
        print(f"SAR Desc Shape: {sample['sar_desc'].shape} (Expected T_sar x 2)")
        
        # Check values
        if sample['bands'].shape[0] > 0:
             print(f"First timestep bands: {sample['bands'][0]}")
    else:
        print("Dataset is empty.")

