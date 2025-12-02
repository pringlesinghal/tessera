#!/usr/bin/env python3
"""
Filter completed_tiles.txt to remove tiles that have validation errors.

This script reads:
1. completed_tiles.txt - list of all completed tiles
2. validation_report_completed.txt - validation report with CSV section

And creates:
- completed_valid_tiles.txt - only tiles with no errors in any year

Usage:
    python filter_valid_tiles.py
"""

from pathlib import Path

# File paths
COMPLETED_TILES_FILE = Path("shapefiles/india_tiles/mgrs_tiles/completed_tiles.txt")
VALIDATION_REPORT_FILE = Path("validation_report_completed.txt")
OUTPUT_FILE = Path("shapefiles/india_tiles/mgrs_tiles/completed_valid_tiles.txt")


def parse_validation_csv(report_path):
    """
    Parse the CSV section of the validation report.
    
    Returns:
        set: Set of tile_ids that have errors
    """
    invalid_tiles = set()
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find the CSV section
    if "INVALID TILES CSV FORMAT" not in content:
        print("No CSV section found in validation report")
        return invalid_tiles
    
    # Extract CSV section
    csv_section = content.split("INVALID TILES CSV FORMAT")[1]
    
    # Parse CSV lines
    for line in csv_section.split('\n'):
        line = line.strip()
        if not line or '=' in line or 'tile_id,year' in line:
            continue
        
        parts = line.split(',')
        if len(parts) == 2:
            tile_id = parts[0].strip()
            invalid_tiles.add(tile_id)
    
    return invalid_tiles


def load_completed_tiles(completed_tiles_path):
    """Load the list of completed tiles."""
    with open(completed_tiles_path, 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]
    return tiles


def main():
    print("=" * 80)
    print("FILTERING VALID TILES")
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
    invalid_tiles = parse_validation_csv(VALIDATION_REPORT_FILE)
    print(f"Found {len(invalid_tiles)} tiles with errors")
    print()
    
    # Filter out invalid tiles
    valid_tiles = [tile for tile in completed_tiles if tile not in invalid_tiles]
    
    # Print summary
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total completed tiles: {len(completed_tiles)}")
    print(f"Tiles with errors: {len(invalid_tiles)}")
    print(f"Valid tiles: {len(valid_tiles)}")
    print(f"Removed: {len(completed_tiles) - len(valid_tiles)} tiles")
    print(f"Retention rate: {len(valid_tiles)/len(completed_tiles)*100:.2f}%")
    print()
    
    # Write valid tiles to output file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for tile in valid_tiles:
            f.write(f"{tile}\n")
    
    print(f"Valid tiles written to: {OUTPUT_FILE}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
