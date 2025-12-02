#!/usr/bin/env python3
"""
Validate tile dimensions across the dataset.

This script iterates through all tiles in the data directory and checks:
1. Dimensions of bands.npy, masks.npy, doys.npy
2. Dimensions of SAR files (if present)
3. Reports any inconsistencies or corrupted files

Usage:
    python validate_tile_dimensions.py --data_dir /path/to/time_series --output validation_report.txt
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_tile(tile_path):
    """
    Validate a single tile's dimensions.
    
    Returns:
        dict: Validation results with keys:
            - 'valid': bool
            - 'errors': list of error messages
            - 'dimensions': dict of file dimensions
    """
    result = {
        'valid': True,
        'errors': [],
        'dimensions': {}
    }
    
    data_path = tile_path / "data_processed"
    
    if not data_path.exists():
        result['valid'] = False
        result['errors'].append(f"Missing data_processed directory")
        return result
    
    # Check S2 files
    required_s2_files = ['bands.npy', 'masks.npy', 'doys.npy']
    for filename in required_s2_files:
        filepath = data_path / filename
        if not filepath.exists():
            result['valid'] = False
            result['errors'].append(f"Missing {filename}")
            continue
        
        try:
            # Try to load with mmap to check file integrity
            arr = np.load(filepath, mmap_mode='r')
            result['dimensions'][filename] = arr.shape
            
            # Validate shape
            if filename == 'bands.npy':
                if len(arr.shape) != 4:
                    result['valid'] = False
                    result['errors'].append(f"{filename} has wrong ndim: {len(arr.shape)} (expected 4)")
                else:
                    # Check spatial dimensions are 2000x2000
                    _, H, W, _ = arr.shape
                    if H != 2000 or W != 2000:
                        result['valid'] = False
                        result['errors'].append(f"{filename} has wrong spatial dimensions: {H}x{W} (expected 2000x2000)")
            elif filename == 'masks.npy':
                if len(arr.shape) != 3:
                    result['valid'] = False
                    result['errors'].append(f"{filename} has wrong ndim: {len(arr.shape)} (expected 3)")
                else:
                    # Check spatial dimensions are 2000x2000
                    _, H, W = arr.shape
                    if H != 2000 or W != 2000:
                        result['valid'] = False
                        result['errors'].append(f"{filename} has wrong spatial dimensions: {H}x{W} (expected 2000x2000)")
            elif filename == 'doys.npy':
                if len(arr.shape) != 1:
                    result['valid'] = False
                    result['errors'].append(f"{filename} has wrong ndim: {len(arr.shape)} (expected 1)")
                    
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Error loading {filename}: {e}")
    
    # Check S1 files (optional)
    s1_files = [
        'sar_ascending.npy', 'sar_ascending_doy.npy',
        'sar_descending.npy', 'sar_descending_doy.npy'
    ]
    
    for filename in s1_files:
        filepath = data_path / filename
        if filepath.exists():
            try:
                arr = np.load(filepath, mmap_mode='r')
                result['dimensions'][filename] = arr.shape
                
                # Validate shape
                if 'doy' not in filename:
                    if len(arr.shape) != 4:
                        result['valid'] = False
                        result['errors'].append(f"{filename} has wrong ndim: {len(arr.shape)} (expected 4)")
                    else:
                        # Check spatial dimensions are 2000x2000
                        _, H, W, _ = arr.shape
                        if H != 2000 or W != 2000:
                            result['valid'] = False
                            result['errors'].append(f"{filename} has wrong spatial dimensions: {H}x{W} (expected 2000x2000)")
                else:
                    if len(arr.shape) != 1:
                        result['valid'] = False
                        result['errors'].append(f"{filename} has wrong ndim: {len(arr.shape)} (expected 1)")
                        
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f"Error loading {filename}: {e}")
    
    # Check dimension consistency
    if 'bands.npy' in result['dimensions'] and 'masks.npy' in result['dimensions']:
        bands_shape = result['dimensions']['bands.npy']
        masks_shape = result['dimensions']['masks.npy']
        
        # bands: (T, H, W, C), masks: (T, H, W)
        if bands_shape[:3] != masks_shape:
            result['valid'] = False
            result['errors'].append(f"Shape mismatch: bands {bands_shape[:3]} vs masks {masks_shape}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate tile dimensions")
    parser.add_argument('--data_dir', type=str, required=True, help="Root data directory")
    parser.add_argument('--output', type=str, default='validation_report.txt', help="Output report file")
    parser.add_argument('--years', type=str, nargs='+', help="Specific years to check (default: all)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    # Determine years to check
    if args.years:
        years = args.years
    else:
        years = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    logger.info(f"Checking years: {years}")
    
    # Statistics
    stats = {
        'total_tiles': 0,
        'valid_tiles': 0,
        'invalid_tiles': 0,
        'dimension_summary': defaultdict(int),
        'errors_by_type': defaultdict(int)
    }
    
    invalid_tiles = []
    
    # Iterate through years and tiles
    for year in years:
        year_dir = data_dir / year
        if not year_dir.exists():
            logger.warning(f"Year directory does not exist: {year_dir}")
            continue
        
        logger.info(f"Checking year {year}...")
        
        tile_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
        
        for tile_dir in tile_dirs:
            stats['total_tiles'] += 1
            
            result = validate_tile(tile_dir)
            
            if result['valid']:
                stats['valid_tiles'] += 1
                
                # Track dimension distribution
                if 'bands.npy' in result['dimensions']:
                    shape = result['dimensions']['bands.npy']
                    dim_key = f"{shape[1]}x{shape[2]}"  # H x W
                    stats['dimension_summary'][dim_key] += 1
            else:
                stats['invalid_tiles'] += 1
                invalid_tiles.append({
                    'year': year,
                    'tile_id': tile_dir.name,
                    'errors': result['errors'],
                    'dimensions': result['dimensions']
                })
                
                # Track error types
                for error in result['errors']:
                    error_type = error.split(':')[0]
                    stats['errors_by_type'][error_type] += 1
            
            # Log progress
            if stats['total_tiles'] % 100 == 0:
                logger.info(f"Processed {stats['total_tiles']} tiles...")
    
    # Write report
    logger.info(f"Writing report to {args.output}")
    
    with open(args.output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TILE DIMENSION VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data Directory: {data_dir}\n")
        f.write(f"Years Checked: {', '.join(years)}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Tiles: {stats['total_tiles']}\n")
        f.write(f"Valid Tiles: {stats['valid_tiles']}\n")
        f.write(f"Invalid Tiles: {stats['invalid_tiles']}\n")
        f.write(f"Success Rate: {stats['valid_tiles']/stats['total_tiles']*100:.2f}%\n\n")
        
        f.write("DIMENSION DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for dim, count in sorted(stats['dimension_summary'].items(), key=lambda x: -x[1]):
            f.write(f"{dim}: {count} tiles\n")
        f.write("\n")
        
        f.write("ERROR TYPES\n")
        f.write("-" * 80 + "\n")
        for error_type, count in sorted(stats['errors_by_type'].items(), key=lambda x: -x[1]):
            f.write(f"{error_type}: {count} occurrences\n")
        f.write("\n")
        
        if invalid_tiles:
            f.write("INVALID TILES DETAILS\n")
            f.write("-" * 80 + "\n")
            for tile_info in invalid_tiles[:100]:  # Limit to first 100
                f.write(f"\nYear: {tile_info['year']}, Tile: {tile_info['tile_id']}\n")
                f.write(f"Dimensions: {tile_info['dimensions']}\n")
                f.write(f"Errors:\n")
                for error in tile_info['errors']:
                    f.write(f"  - {error}\n")
            
            if len(invalid_tiles) > 100:
                f.write(f"\n... and {len(invalid_tiles) - 100} more invalid tiles\n")
    
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Tiles: {stats['total_tiles']}")
    logger.info(f"Valid: {stats['valid_tiles']} ({stats['valid_tiles']/stats['total_tiles']*100:.2f}%)")
    logger.info(f"Invalid: {stats['invalid_tiles']} ({stats['invalid_tiles']/stats['total_tiles']*100:.2f}%)")
    logger.info(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
