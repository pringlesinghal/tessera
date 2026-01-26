#!/usr/bin/env python3
"""
Quick test validation script - validates a small subset to verify the approach works.
Run this first before the full validation to catch any issues.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_tile_data_check(tile_path_str):
    """Fast tile validation - check if all required files exist and can be loaded."""
    try:
        tile_path = Path(tile_path_str)
        if not tile_path.exists():
            return False
            
        required_files = ["bands.npy", "masks.npy", "doys.npy"]
        for fname in required_files:
            fpath = tile_path / fname
            if not fpath.exists():
                return False
            try:
                arr = np.load(fpath, mmap_mode='r')
                shape = arr.shape  # This will fail if file is corrupted
                del arr
            except:
                return False
                
        return True
    except Exception:
        return False

def test_validation():
    """Test validation on a small subset of data."""
    
    # Configuration for test
    INDEX_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled"
    DATA_DIR = os.environ.get('SCRATCH', '/scratch/users/psinghal') + "/time_series"
    YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  # Test with all years
    
    logger.info("=== Quick Validation Test ===")
    logger.info(f"Index directory: {INDEX_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Test years: {YEARS}")
    
    # Read first parquet file to get sample tiles
    test_file = Path(INDEX_DIR) / "part.0.parquet"
    logger.info(f"Reading test file: {test_file}")
    
    df = pd.read_parquet(test_file)
    logger.info(f"Test file contains {len(df):,} samples")
    
    # Get first 50 unique tiles for testing
    unique_tiles = df['tile_id'].unique()[:50]
    logger.info(f"Testing first {len(unique_tiles)} unique tiles")
    
    # Test validation
    valid_count = 0
    invalid_count = 0
    
    for tile_id in tqdm(unique_tiles, desc="Testing tiles"):
        is_valid = True
        for year in YEARS:
            tile_path = Path(DATA_DIR) / str(year) / tile_id / "data_processed"
            if not load_tile_data_check(str(tile_path)):
                is_valid = False
                break
        
        if is_valid:
            valid_count += 1
            logger.info(f"✓ VALID: {tile_id}")
        else:
            invalid_count += 1
            logger.info(f"✗ INVALID: {tile_id}")
    
    total = valid_count + invalid_count
    valid_pct = (valid_count / total) * 100 if total > 0 else 0
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Total tiles tested: {total}")
    logger.info(f"Valid tiles: {valid_count} ({valid_pct:.1f}%)")
    logger.info(f"Invalid tiles: {invalid_count} ({100-valid_pct:.1f}%)")
    
    if invalid_count > 0:
        logger.info(f"Found {invalid_count} invalid tiles - full validation will filter these out")
    else:
        logger.info("All test tiles are valid!")
    
    # Test sample filtering
    logger.info(f"\n=== Testing Sample Filtering ===")
    if valid_count > 0:
        # Get valid tile IDs
        valid_tiles = set()
        for tile_id in unique_tiles:
            is_valid = True
            for year in YEARS:
                tile_path = Path(DATA_DIR) / str(year) / tile_id / "data_processed"
                if not load_tile_data_check(str(tile_path)):
                    is_valid = False
                    break
            if is_valid:
                valid_tiles.add(tile_id)
        
        # Filter test data
        df_filtered = df[df['tile_id'].isin(valid_tiles)]
        original_samples = len(df)
        filtered_samples = len(df_filtered)
        retention_pct = (filtered_samples / original_samples) * 100
        
        logger.info(f"Original samples in {test_file.name}: {original_samples:,}")
        logger.info(f"Filtered samples: {filtered_samples:,}")
        logger.info(f"Retention rate: {retention_pct:.1f}%")
    
    logger.info(f"\n=== Test Complete ===")
    logger.info("If results look good, run the full validation with:")
    logger.info("sbatch validate_dataset.sbatch")

if __name__ == "__main__":
    test_validation()