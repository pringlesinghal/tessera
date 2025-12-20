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


class TreeSpeciesValidationDataset(Dataset):
    """
    Dataset for labeled tree species validation.
    
    Structure:
    - Coordinates: CSV file with lat/long coordinates mapped to tile IDs and pixel positions
    - Data: Numpy files (bands.npy, etc.) stored in {data_dir}/{year}/{tile_id}/data_processed/
    - Labels: Tree species names from the CSV file

    Strategy: "Direct Coordinate Loading"
    - Loads labeled coordinates from mapped CSV files
    - Extracts pixel timeseries for specific coordinates
    - Returns features with species labels for supervised learning
    """

    def __init__(self, mapped_csv_path, data_dir, year=None, years=None, cache_size=50,
                 sample_size_s2=20, sample_size_s1=20, normalize=True):
        """
        Args:
            mapped_csv_path: Path to CSV file with mapped coordinates and species labels
            data_dir: Root directory containing year subdirectories with tile data
            year: If specified, only use this year. Otherwise random.
            years: If specified, list of years to sample from. Otherwise all available.
            cache_size: Number of tiles to keep in memory
            sample_size_s2: Number of timesteps to sample for S2
            sample_size_s1: Number of timesteps to sample for S1
            normalize: Whether to normalize the data
        """
        
        self.mapped_csv_path = Path(mapped_csv_path)
        self.data_dir = Path(data_dir)
        self.year = year
        self.years = years if years else [2019, 2020, 2021, 2022, 2023]  # Focus on recent years
        self.cache_size = cache_size
        self.sample_size_s2 = sample_size_s2
        self.sample_size_s1 = sample_size_s1
        self.normalize = normalize

        # 1. Load Mapped Coordinates
        print(f"Loading mapped coordinates from {self.mapped_csv_path}...")
        self.coordinates_df = pd.read_csv(self.mapped_csv_path)
        
        # Filter out failed mappings
        valid_mappings = self.coordinates_df['tile_id'].notna()
        self.coordinates_df = self.coordinates_df[valid_mappings].reset_index(drop=True)
        print(f"Loaded {len(self.coordinates_df)} valid coordinate mappings")
        
        # 2. Create Species Label Mapping
        unique_species = sorted(self.coordinates_df['class'].unique())
        self.species_to_id = {species: idx for idx, species in enumerate(unique_species)}
        self.id_to_species = {idx: species for species, idx in self.species_to_id.items()}
        print(f"Found {len(unique_species)} unique species")
        
        # 3. Determine split and load/create species groupings
        self.split = self._infer_split_from_path(self.mapped_csv_path)
        self.species_counts = self.coordinates_df['class'].value_counts().to_dict()
        self._load_or_create_species_groups()
        
        self.total_length = len(self.coordinates_df)
        
        # 3. Cache State - same as original
        
        # Tile Data Cache: { "tile_id_year": { "bands": memmap, ... } }
        self.tile_cache = {}
        self.cache_queue = [] # For LRU eviction
        
        # 4. Corruption Handling
        self.corrupted_tiles = set()  # Cache of known corrupted tile-year combinations
        self.corruption_stats = {
            'total_attempts': 0,
            'corrupted_hits': 0,
            'cache_saves': 0,
            'retries': 0
        }

    def _infer_split_from_path(self, csv_path):
        """Infer dataset split (train/val/test) from CSV filename."""
        filename = Path(csv_path).stem
        
        # Check for mapped CSV patterns: train_mapped.csv, val_mapped.csv, test_mapped.csv
        if filename == 'train_mapped':
            return 'train'
        elif filename == 'val_mapped':
            return 'val'
        elif filename == 'test_mapped':
            return 'test'
        else:
            raise ValueError(f"Cannot infer split from filename '{filename}'. "
                           f"Expected 'train_mapped.csv', 'val_mapped.csv', or 'test_mapped.csv'")
    
    def _get_species_groups_cache_path(self):
        """Get path for cached species groups file."""
        csv_dir = self.mapped_csv_path.parent
        return csv_dir / "species_groups_cache.json"
    
    def _load_or_create_species_groups(self):
        """Load existing species groups or create new ones (train only)."""
        cache_path = self._get_species_groups_cache_path()
        
        if self.split == 'train':
            # Training set: create new grouping and save
            print(f"Training split detected - creating species groups based on training distribution")
            self._create_species_groups()
            self._save_species_groups(cache_path)
        else:
            # Val/test set: load from cache
            if cache_path.exists():
                print(f"{self.split.title()} split detected - loading species groups from cache")
                self._load_species_groups(cache_path)
            else:
                raise FileNotFoundError(f"No cached species groups found for {self.split} split at {cache_path}. "
                                      f"Please run with train_mapped.csv first to create species groups.")
    
    def _save_species_groups(self, cache_path):
        """Save species groups to cache file."""
        import json
        
        cache_data = {
            'dominant_species': list(self.dominant_species),
            'long_tail_species': list(self.long_tail_species),
            'species_groups': self.species_groups,
            'created_from': 'train',
            'total_train_samples': len(self.coordinates_df),
            'dominant_train_samples': sum(self.species_counts[s] for s in self.dominant_species),
            'long_tail_train_samples': sum(self.species_counts[s] for s in self.long_tail_species)
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Species groups cached to: {cache_path}")
    
    def _load_species_groups(self, cache_path):
        """Load species groups from cache file."""
        import json
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Failed to load species groups cache from {cache_path}: {e}")
        
        self.dominant_species = set(cache_data['dominant_species'])
        self.long_tail_species = set(cache_data['long_tail_species'])
        self.species_groups = cache_data['species_groups']
        
        # Validate that all species in current split are covered
        current_species = set(self.coordinates_df['class'].unique())
        cached_species = self.dominant_species | self.long_tail_species
        
        missing_species = current_species - cached_species
        if missing_species:
            raise ValueError(f"Species {missing_species} in {self.split} split not found in cached groups. "
                           f"Please recreate cache with updated training data.")
        
        # Calculate stats for current split
        dominant_samples = sum(self.species_counts.get(s, 0) for s in self.dominant_species)
        long_tail_samples = sum(self.species_counts.get(s, 0) for s in self.long_tail_species)
        total_samples = len(self.coordinates_df)
        
        print(f"Loaded species groups (from training set):")
        print(f"  Dominant species ({len(self.dominant_species)}): {list(self.dominant_species)}")
        print(f"  Long tail species ({len(self.long_tail_species)}): {list(self.long_tail_species)}")
        print(f"  {self.split} split distribution:")
        print(f"    Dominant group: {dominant_samples} samples ({dominant_samples/total_samples*100:.1f}%)")
        print(f"    Long tail group: {long_tail_samples} samples ({long_tail_samples/total_samples*100:.1f}%)")

    def _create_species_groups(self):
        """
        Create species groups based on cumulative sample distribution.
        Dominant species: top species that make up 75% of samples
        Long tail species: bottom species that make up 25% of samples
        """
        total_samples = len(self.coordinates_df)
        threshold_samples = 0.75 * total_samples
        
        # Sort species by count (descending)
        sorted_species = sorted(self.species_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Accumulate samples from largest to smallest until we hit 75%
        dominant_species = []
        long_tail_species = []
        cumulative_samples = 0
        
        for species, count in sorted_species:
            if cumulative_samples < threshold_samples:
                dominant_species.append(species)
                cumulative_samples += count
            else:
                long_tail_species.append(species)
        
        self.dominant_species = set(dominant_species)
        self.long_tail_species = set(long_tail_species)
        
        # Create group mappings
        self.species_groups = {}
        for species in dominant_species:
            self.species_groups[species] = 'dominant'
        for species in long_tail_species:
            self.species_groups[species] = 'long_tail'
        
        # Calculate actual percentages
        dominant_samples = sum(self.species_counts[s] for s in dominant_species)
        long_tail_samples = sum(self.species_counts[s] for s in long_tail_species)
        
        print(f"Species grouping (75/25 split):")
        print(f"  Dominant species ({len(dominant_species)}): {dominant_species}")
        print(f"  Long tail species ({len(long_tail_species)}): {long_tail_species}")
        print(f"  Dominant group: {dominant_samples} samples ({dominant_samples/total_samples*100:.1f}%)")
        print(f"  Long tail group: {long_tail_samples} samples ({long_tail_samples/total_samples*100:.1f}%)")

    def _get_coordinate_info(self, idx):
        """Get coordinate information for a specific sample index."""
        return self.coordinates_df.iloc[idx]

    def _get_tile_data(self, tile_id, year):
        """
        Retrieves tile data arrays from cache or disk with fast corruption handling.
        Returns dictionary of memmap arrays.
        """
        t0 = time.time()
        cache_key = f"{tile_id}_{year}"
        
        # Fast corruption check - avoid known bad tiles immediately
        if cache_key in self.corrupted_tiles:
            self.corruption_stats['cache_saves'] += 1
            raise FileNotFoundError(f"Known corrupted tile: {cache_key}")

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

        # Load new tile with timeout protection
        tile_path = self.data_dir / str(year) / tile_id / "data_processed"

        if not tile_path.exists():
            # Cache this corruption and fail fast
            self.corrupted_tiles.add(cache_key)
            raise FileNotFoundError(f"Tile data not found: {tile_path}")

        try:
            arrays = {}
            # S2 Data (Required) - load with minimal validation
            arrays["bands"] = np.load(tile_path / "bands.npy", mmap_mode="r")
            arrays["masks"] = np.load(tile_path / "masks.npy", mmap_mode="r")
            arrays["doys"] = np.load(tile_path / "doys.npy", mmap_mode="r")
            
            # Quick validation - just check shapes are accessible
            _ = arrays["bands"].shape
            _ = arrays["masks"].shape
            _ = arrays["doys"].shape
            
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
            # Cache this corruption for future fast-fail
            self.corrupted_tiles.add(cache_key)
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

    def __getitem__(self, idx, year_override=None):
        """Get labeled tree species sample with features and species label."""
        self.corruption_stats['total_attempts'] += 1
        
        try:
            # Get coordinate information
            coord_info = self._get_coordinate_info(idx)
            
            # Extract metadata
            species_name = coord_info["class"]
            species_id = self.species_to_id[species_name]
            tile_id = coord_info["tile_id"]
            pixel_row = int(coord_info["pixel_row"]) 
            pixel_col = int(coord_info["pixel_col"])
            
            # Determine year to use
            if year_override is not None:
                target_year = year_override
            elif self.year is not None:
                target_year = self.year
            else:
                # Try to find data from any available year
                for year in self.years:
                    cache_key = f"{tile_id}_{year}"
                    if cache_key not in self.corrupted_tiles:
                        target_year = year
                        break
                else:
                    target_year = int(np.random.choice(self.years))
            
            # Fast corruption check before attempting load
            cache_key = f"{tile_id}_{target_year}"
            if cache_key in self.corrupted_tiles:
                self.corruption_stats['corrupted_hits'] += 1
                raise FileNotFoundError(f"Known corrupted: {cache_key}")

            # Load tile data
            tile_data = self._get_tile_data(tile_id, target_year)
            
            # Process sample and add species label
            result = self._process_sample(tile_data, pixel_row, pixel_col, target_year, idx)
            
            # Add species information to result
            result["species_label"] = species_name
            result["species_id"] = species_id
            result["tile_id"] = tile_id
            result["pixel_row"] = pixel_row
            result["pixel_col"] = pixel_col
            
            return result
            
        except (ValueError, RuntimeError, OSError, FileNotFoundError) as e:
            # Handle corruption by returning None (caller should handle)
            logger.warning(f"Failed to load sample {idx}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error on sample {idx}: {e}")
            return None
    
    def _process_sample(self, tile_data, row, col, target_year, global_idx):
        """Process a successfully loaded tile into a training sample."""
        t_total = time.time()
        
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

    def get_corruption_stats(self):
        """Get statistics about corruption handling."""
        stats = self.corruption_stats.copy()
        stats['corrupted_tiles_cached'] = len(self.corrupted_tiles)
        if stats['total_attempts'] > 0:
            stats['corruption_rate'] = stats['corrupted_hits'] / stats['total_attempts']
            stats['retry_rate'] = stats['retries'] / stats['total_attempts']
        else:
            stats['corruption_rate'] = 0.0
            stats['retry_rate'] = 0.0
        return stats

    def clear_corruption_cache(self):
        """Clear the corruption cache (useful for testing)."""
        self.corrupted_tiles.clear()
        self.corruption_stats = {
            'total_attempts': 0,
            'corrupted_hits': 0,
            'cache_saves': 0,
            'retries': 0
        }
    
    def get_species_info(self):
        """Get comprehensive species information for evaluation."""
        return {
            'species_to_id': self.species_to_id,
            'id_to_species': self.id_to_species,
            'species_counts': self.species_counts,
            'species_groups': self.species_groups,
            'dominant_species': list(self.dominant_species),
            'long_tail_species': list(self.long_tail_species),
            'num_species': len(self.species_to_id)
        }
    
    def get_species_group_ids(self):
        """Get species IDs grouped by dominant/long_tail for evaluation."""
        dominant_ids = [self.species_to_id[species] for species in self.dominant_species]
        long_tail_ids = [self.species_to_id[species] for species in self.long_tail_species]
        
        return {
            'dominant': dominant_ids,
            'long_tail': long_tail_ids
        }
    
    def filter_by_species_group(self, group='dominant'):
        """
        Create a filtered dataset containing only samples from specified group.
        
        Args:
            group: 'dominant' or 'long_tail'
            
        Returns:
            Filtered TreeSpeciesValidationDataset
        """
        if group == 'dominant':
            target_species = self.dominant_species
        elif group == 'long_tail':
            target_species = self.long_tail_species
        else:
            raise ValueError(f"Invalid group: {group}. Must be 'dominant' or 'long_tail'")
        
        # Filter dataframe
        mask = self.coordinates_df['class'].isin(target_species)
        filtered_df = self.coordinates_df[mask].reset_index(drop=True)
        
        # Create new dataset with filtered data
        filtered_dataset = TreeSpeciesValidationDataset.__new__(TreeSpeciesValidationDataset)
        
        # Copy attributes
        filtered_dataset.mapped_csv_path = self.mapped_csv_path
        filtered_dataset.data_dir = self.data_dir
        filtered_dataset.year = self.year
        filtered_dataset.years = self.years
        filtered_dataset.cache_size = self.cache_size
        filtered_dataset.sample_size_s2 = self.sample_size_s2
        filtered_dataset.sample_size_s1 = self.sample_size_s1
        filtered_dataset.normalize = self.normalize
        
        # Use filtered coordinates
        filtered_dataset.coordinates_df = filtered_df
        filtered_dataset.total_length = len(filtered_df)
        
        # Create species mappings for filtered data
        unique_species = sorted(filtered_df['class'].unique())
        filtered_dataset.species_to_id = {species: idx for idx, species in enumerate(unique_species)}
        filtered_dataset.id_to_species = {idx: species for species, idx in filtered_dataset.species_to_id.items()}
        filtered_dataset.species_counts = filtered_df['class'].value_counts().to_dict()
        
        # Initialize cache and corruption tracking
        filtered_dataset.tile_cache = {}
        filtered_dataset.cache_queue = []
        filtered_dataset.corrupted_tiles = set()
        filtered_dataset.corruption_stats = {
            'total_attempts': 0,
            'corrupted_hits': 0,
            'cache_saves': 0,
            'retries': 0
        }
        
        return filtered_dataset


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--mapped_csv", 
                       default="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/labelled_tree_data/val_mapped.csv",
                       help="Path to mapped CSV file")
    parser.add_argument("--data_dir", 
                       default="/scratch/groups/dlobell/shared_data",
                       help="Root data directory")
    parser.add_argument("--years", nargs='+', type=int,
                       default=[2022, 2021, 2020],
                       help="Years to try loading")
    args = parser.parse_args()

    print("\n--- Initializing TreeSpeciesValidationDataset ---")
    ds = TreeSpeciesValidationDataset(args.mapped_csv, args.data_dir, years=args.years)

    print(f"Dataset length: {len(ds)}")
    print(f"Species mapping:")
    for species, idx in ds.species_to_id.items():
        count = (ds.coordinates_df['class'] == species).sum()
        print(f"  {idx:2d}: {species} ({count} samples)")

    if len(ds) > 0:
        print("\n--- Loading Sample Tests ---")
        
        # Test a few samples
        for test_idx in [0, 10, 50]:
            if test_idx >= len(ds):
                continue
                
            print(f"\nTesting sample {test_idx}:")
            coord_info = ds._get_coordinate_info(test_idx)
            print(f"  Species: {coord_info['class']}")
            print(f"  Tile: {coord_info['tile_id']}")
            print(f"  Pixel: ({coord_info['pixel_row']}, {coord_info['pixel_col']})")
            
            start_t = time.time()
            sample = ds[test_idx]
            dur = time.time() - start_t
            
            if sample is not None:
                print(f"  Loaded in {dur:.4f} sec")
                print(f"  Species ID: {sample['species_id']} ({sample['species_label']})")
                for k, v in sample.items():
                    if torch.is_tensor(v) and v.numel() > 0:
                        print(f"  {k}: {v.shape}, dtype={v.dtype}, mean={v.mean():.3f}")
                    elif isinstance(v, (str, int)):
                        print(f"  {k}: {v}")
            else:
                print(f"  Failed to load (no data available)")

        print(f"\n--- Dataset Statistics ---")
        stats = ds.get_corruption_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Dataset is empty.")
