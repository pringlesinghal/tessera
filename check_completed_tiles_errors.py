#!/usr/bin/env python3
"""
Check which tiles from completed_tiles.txt have errors in the validation report.

Usage:
    python check_completed_tiles_errors.py
"""

import re
from pathlib import Path

# File paths
COMPLETED_TILES_FILE = Path("shapefiles/india_tiles/mgrs_tiles/completed_tiles.txt")
VALIDATION_REPORT_FILE = Path("validation_report.txt")
OUTPUT_FILE = Path("tiles_with_errors.txt")

def parse_validation_report(report_path):
    """
    Parse the validation report and extract tiles with errors.
    
    Returns:
        dict: {tile_id: {'year': year, 'errors': [error_list], 'dimensions': dimensions}}
    """
    tiles_with_errors = {}
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find the "INVALID TILES DETAILS" section
    if "INVALID TILES DETAILS" not in content:
        print("No invalid tiles found in validation report")
        return tiles_with_errors
    
    # Extract the invalid tiles section
    invalid_section = content.split("INVALID TILES DETAILS")[1]
    
    # Parse each tile entry
    # Format: "Year: YYYY, Tile: tile_id"
    tile_entries = re.split(r'\n(?=Year: )', invalid_section)
    
    for entry in tile_entries:
        if not entry.strip():
            continue
            
        # Extract year and tile_id
        year_match = re.search(r'Year: (\d+)', entry)
        tile_match = re.search(r'Tile: ([^\n]+)', entry)
        
        if year_match and tile_match:
            year = year_match.group(1)
            tile_id = tile_match.group(1).strip()
            
            # Extract errors
            errors = []
            if "Errors:" in entry:
                error_section = entry.split("Errors:")[1]
                error_lines = [line.strip() for line in error_section.split('\n') if line.strip().startswith('-')]
                errors = [line.lstrip('- ').strip() for line in error_lines]
            
            # Extract dimensions
            dimensions = None
            if "Dimensions:" in entry:
                dim_match = re.search(r'Dimensions: ({[^}]+})', entry)
                if dim_match:
                    dimensions = dim_match.group(1)
            
            tiles_with_errors[tile_id] = {
                'year': year,
                'errors': errors,
                'dimensions': dimensions
            }
    
    return tiles_with_errors


def load_completed_tiles(completed_tiles_path):
    """Load the list of completed tiles."""
    with open(completed_tiles_path, 'r') as f:
        # Each line is a tile ID
        tiles = [line.strip() for line in f if line.strip()]
    return set(tiles)


def main():
    print("=" * 80)
    print("CHECKING COMPLETED TILES FOR ERRORS")
    print("=" * 80)
    print()
    
    # Check if files exist
    if not COMPLETED_TILES_FILE.exists():
        print(f"ERROR: Completed tiles file not found: {COMPLETED_TILES_FILE}")
        return
    
    if not VALIDATION_REPORT_FILE.exists():
        print(f"ERROR: Validation report not found: {VALIDATION_REPORT_FILE}")
        return
    
    # Load data
    print(f"Loading completed tiles from: {COMPLETED_TILES_FILE}")
    completed_tiles = load_completed_tiles(COMPLETED_TILES_FILE)
    print(f"Found {len(completed_tiles)} completed tiles")
    print()
    
    print(f"Parsing validation report from: {VALIDATION_REPORT_FILE}")
    tiles_with_errors = parse_validation_report(VALIDATION_REPORT_FILE)
    print(f"Found {len(tiles_with_errors)} tiles with errors in validation report")
    print()
    
    # Find intersection
    completed_with_errors = {}
    for tile_id in completed_tiles:
        if tile_id in tiles_with_errors:
            completed_with_errors[tile_id] = tiles_with_errors[tile_id]
    
    # Print summary
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Completed tiles: {len(completed_tiles)}")
    print(f"Tiles with errors: {len(tiles_with_errors)}")
    print(f"Completed tiles WITH errors: {len(completed_with_errors)}")
    print(f"Percentage of completed tiles with errors: {len(completed_with_errors)/len(completed_tiles)*100:.2f}%")
    print()
    
    # Write detailed report
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPLETED TILES WITH ERRORS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total completed tiles: {len(completed_tiles)}\n")
        f.write(f"Completed tiles with errors: {len(completed_with_errors)}\n")
        f.write(f"Error rate: {len(completed_with_errors)/len(completed_tiles)*100:.2f}%\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("DETAILS\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by error type
        error_type_counts = {}
        for tile_id, info in completed_with_errors.items():
            for error in info['errors']:
                error_type = error.split(':')[0] if ':' in error else error
                if error_type not in error_type_counts:
                    error_type_counts[error_type] = []
                error_type_counts[error_type].append(tile_id)
        
        f.write("ERROR TYPE SUMMARY:\n")
        f.write("-" * 80 + "\n")
        for error_type, tile_list in sorted(error_type_counts.items(), key=lambda x: -len(x[1])):
            f.write(f"{error_type}: {len(tile_list)} tiles\n")
        f.write("\n")
        
        f.write("DETAILED LIST:\n")
        f.write("-" * 80 + "\n")
        for tile_id in sorted(completed_with_errors.keys()):
            info = completed_with_errors[tile_id]
            f.write(f"\nTile: {tile_id}\n")
            f.write(f"Year: {info['year']}\n")
            if info['dimensions']:
                f.write(f"Dimensions: {info['dimensions']}\n")
            f.write(f"Errors:\n")
            for error in info['errors']:
                f.write(f"  - {error}\n")
    
    print(f"Detailed report written to: {OUTPUT_FILE}")
    print()
    
    # Print sample of problematic tiles
    if completed_with_errors:
        print("SAMPLE OF TILES WITH ERRORS (first 10):")
        print("-" * 80)
        for i, (tile_id, info) in enumerate(list(completed_with_errors.items())[:10]):
            print(f"{i+1}. {tile_id} (Year: {info['year']})")
            for error in info['errors']:
                print(f"   - {error}")
        
        if len(completed_with_errors) > 10:
            print(f"\n... and {len(completed_with_errors) - 10} more tiles with errors")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
