#!/usr/bin/env python3
"""
Optimized Dataset Validation Script

Efficiently validates 200M+ tree pixels across ~80TB of data to create
a filtered parquet index containing only samples from tiles that are valid
across ALL years (2016-2024).

Strategy:
1. Extract unique tiles from all parquet index files
2. Test each tile across all years to identify completely valid tiles  
3. Filter original parquet files to keep only samples from valid tiles
4. Write new filtered parquet index

Memory efficient: processes tiles in batches, doesn't load full dataset into memory.
"""

import os
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_tile_data_check(tile_path_str):
    """
    Fast tile validation - check if all required files exist and can be loaded.
    Returns True if tile is valid, False otherwise.
    """
    try:
        tile_path = Path(tile_path_str)
        
        # Check if directory exists
        if not tile_path.exists():
            return False
            
        # Check required files exist
        required_files = ["bands.npy", "masks.npy", "doys.npy"]
        for fname in required_files:
            fpath = tile_path / fname
            if not fpath.exists():
                return False
            # Quick file integrity check - try to get shape without loading data
            try:
                arr = np.load(fpath, mmap_mode='r')
                shape = arr.shape  # This will fail if file is corrupted
                del arr  # Release memory map
            except:
                return False
                
        return True
        
    except Exception:
        return False

def validate_tile_all_years(args):
    """
    Validate a single tile across all years.
    Returns (tile_id, is_valid_all_years)
    """
    tile_id, data_dir, years = args
    
    try:
        for year in years:
            tile_path = Path(data_dir) / str(year) / tile_id / "data_processed"
            if not load_tile_data_check(str(tile_path)):
                return tile_id, False
        
        return tile_id, True
        
    except Exception as e:
        logger.error(f"Error validating tile {tile_id}: {e}")
        return tile_id, False

def extract_unique_tiles(index_dir):
    """
    Extract all unique tile_ids from parquet index files efficiently.
    """
    logger.info(f"Extracting unique tiles from {index_dir}")
    unique_tiles = set()
    
    parquet_files = sorted([f for f in Path(index_dir).glob("part.*.parquet")])
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    for pq_file in tqdm(parquet_files, desc="Scanning parquet files"):
        try:
            # Read only tile_id column to save memory
            df = pd.read_parquet(pq_file, columns=['tile_id'])
            unique_tiles.update(df['tile_id'].unique())
            logger.info(f"Processed {pq_file.name}, total unique tiles so far: {len(unique_tiles):,}")
        except Exception as e:
            logger.error(f"Error reading {pq_file}: {e}")
            
    logger.info(f"Found {len(unique_tiles):,} unique tiles total")
    return unique_tiles

def validate_tiles_parallel(unique_tiles, data_dir, years, max_workers=None):
    """
    Validate tiles in parallel across all years.
    """
    if max_workers is None:
        max_workers = min(32, mp.cpu_count())
        
    logger.info(f"Validating {len(unique_tiles):,} tiles across years {years} using {max_workers} workers")
    
    # Prepare arguments for parallel processing
    args_list = [(tile_id, data_dir, years) for tile_id in unique_tiles]
    
    valid_tiles = set()
    invalid_tiles = set()
    
    # Process in batches to manage memory
    batch_size = 1000
    total_batches = (len(args_list) + batch_size - 1) // batch_size
    
    with tqdm(total=len(unique_tiles), desc="Validating tiles") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(args_list))
                batch_args = args_list[start_idx:end_idx]
                
                # Submit batch
                futures = [executor.submit(validate_tile_all_years, args) for args in batch_args]
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        tile_id, is_valid = future.result()
                        if is_valid:
                            valid_tiles.add(tile_id)
                        else:
                            invalid_tiles.add(tile_id)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing tile: {e}")
                        pbar.update(1)
                
                # Log progress
                valid_count = len(valid_tiles)
                invalid_count = len(invalid_tiles)
                total_processed = valid_count + invalid_count
                if total_processed > 0:
                    valid_pct = (valid_count / total_processed) * 100
                    logger.info(f"Batch {batch_idx+1}/{total_batches} done. "
                              f"Valid: {valid_count:,} ({valid_pct:.1f}%), "
                              f"Invalid: {invalid_count:,}")
    
    return valid_tiles, invalid_tiles

def filter_parquet_files(index_dir, output_dir, valid_tiles):
    """
    Filter original parquet files to keep only samples from valid tiles.
    """
    logger.info(f"Filtering parquet files, keeping samples from {len(valid_tiles):,} valid tiles")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted([f for f in Path(index_dir).glob("part.*.parquet")])
    total_original_samples = 0
    total_valid_samples = 0
    
    for pq_file in tqdm(parquet_files, desc="Filtering parquet files"):
        try:
            # Read full parquet file
            df = pd.read_parquet(pq_file)
            total_original_samples += len(df)
            
            # Filter to keep only valid tiles
            df_filtered = df[df['tile_id'].isin(valid_tiles)]
            total_valid_samples += len(df_filtered)
            
            # Write filtered file if it has data
            if len(df_filtered) > 0:
                output_file = output_dir / pq_file.name
                df_filtered.to_parquet(output_file, index=False)
                logger.info(f"Filtered {pq_file.name}: {len(df):,} -> {len(df_filtered):,} samples")
            else:
                logger.info(f"Skipped {pq_file.name}: no valid samples")
                
        except Exception as e:
            logger.error(f"Error filtering {pq_file}: {e}")
    
    logger.info(f"Filtering complete. Total samples: {total_original_samples:,} -> {total_valid_samples:,} "
               f"({(total_valid_samples/total_original_samples)*100:.1f}% retained)")
    
    return total_original_samples, total_valid_samples

def main():
    # Configuration
    INDEX_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled"
    DATA_DIR = os.environ.get('SCRATCH', '/scratch/users/psinghal') + "/time_series"
    OUTPUT_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled_valid"
    
    YEARS = list(range(2016, 2025))  # 2016-2024
    MAX_WORKERS = 16  # Adjust based on system capacity
    
    logger.info("=== Dataset Validation Started ===")
    logger.info(f"Index directory: {INDEX_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Years to validate: {YEARS}")
    logger.info(f"Max workers: {MAX_WORKERS}")
    
    start_time = time.time()
    
    # Step 1: Extract unique tiles
    logger.info("\n=== Step 1: Extract Unique Tiles ===")
    unique_tiles = extract_unique_tiles(INDEX_DIR)
    
    # Step 2: Validate tiles across all years
    logger.info("\n=== Step 2: Validate Tiles Across All Years ===")
    valid_tiles, invalid_tiles = validate_tiles_parallel(unique_tiles, DATA_DIR, YEARS, MAX_WORKERS)
    
    # Log validation results
    total_tiles = len(unique_tiles)
    valid_count = len(valid_tiles)
    invalid_count = len(invalid_tiles)
    valid_pct = (valid_count / total_tiles) * 100 if total_tiles > 0 else 0
    
    logger.info(f"\n=== Validation Results ===")
    logger.info(f"Total unique tiles: {total_tiles:,}")
    logger.info(f"Valid tiles (all years): {valid_count:,} ({valid_pct:.1f}%)")
    logger.info(f"Invalid tiles: {invalid_count:,} ({100-valid_pct:.1f}%)")
    
    # Save validation results
    results_file = Path(OUTPUT_DIR) / "validation_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"Dataset Validation Results\n")
        f.write(f"========================\n\n")
        f.write(f"Validation time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Index directory: {INDEX_DIR}\n")
        f.write(f"Data directory: {DATA_DIR}\n")
        f.write(f"Years validated: {YEARS}\n\n")
        f.write(f"Total unique tiles: {total_tiles:,}\n")
        f.write(f"Valid tiles: {valid_count:,} ({valid_pct:.1f}%)\n")
        f.write(f"Invalid tiles: {invalid_count:,} ({100-valid_pct:.1f}%)\n\n")
        f.write(f"Valid tiles list:\n")
        for tile in sorted(valid_tiles):
            f.write(f"{tile}\n")
        f.write(f"\nInvalid tiles list:\n")
        for tile in sorted(invalid_tiles):
            f.write(f"{tile}\n")
    
    # Step 3: Filter parquet files
    logger.info("\n=== Step 3: Filter Parquet Files ===")
    original_samples, valid_samples = filter_parquet_files(INDEX_DIR, OUTPUT_DIR, valid_tiles)
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n=== Final Summary ===")
    logger.info(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Tiles: {total_tiles:,} -> {valid_count:,} ({valid_pct:.1f}% retained)")
    logger.info(f"Samples: {original_samples:,} -> {valid_samples:,} "
               f"({(valid_samples/original_samples)*100:.1f}% retained)")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Validation results: {results_file}")
    logger.info("=== Dataset Validation Complete ===")

if __name__ == "__main__":
    main()