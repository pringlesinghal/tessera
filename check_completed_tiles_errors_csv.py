#!/usr/bin/env python3
"""
Check which tiles from completed_tiles.txt have errors in the validation report.
This version reads the CSV format from the validation report for faster parsing.

Usage:
    python check_completed_tiles_errors_csv.py
"""

from pathlib import Path

# File paths
COMPLETED_TILES_FILE = Path("shapefiles/india_tiles/mgrs_tiles/completed_tiles.txt")
VALIDATION_REPORT_FILE = Path("validation_report.txt")
OUTPUT_FILE = Path("tiles_with_errors.txt")


def parse_validation_csv(report_path):
    """
    Parse the CSV section of the validation report.
    
    Returns:
        dict: {tile_id: [list of years]}
    """
    tiles_with_errors = {}
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find the CSV section
    if "INVALID TILES CSV FORMAT" not in content:
        print("No CSV section found in validation report")
        return tiles_with_errors
    
    # Extract CSV section
    csv_section = content.split("INVALID TILES CSV FORMAT")[1]
    
    # Parse CSV lines
    for line in csv_section.split('\n'):
        line = line.strip()
        if not line or '=' in line or 'tile_id,year' in line:
            continue
        
        parts = line.split(',')
        if len(parts) == 2:
            tile_id, year = parts[0].strip(), parts[1].strip()
            if tile_id not in tiles_with_errors:
                tiles_with_errors[tile_id] = []
            tiles_with_errors[tile_id].append(year)
    
    return tiles_with_errors


def load_completed_tiles(completed_tiles_path):
    """Load the list of completed tiles."""
    with open(completed_tiles_path, 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]
    return set(tiles)


def main():
    print("=" * 80)
    print("CHECKING COMPLETED TILES FOR ERRORS (CSV VERSION)")
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
    
    print(f"Parsing validation report CSV from: {VALIDATION_REPORT_FILE}")
    tiles_with_errors = parse_validation_csv(VALIDATION_REPORT_FILE)
    print(f"Found {len(tiles_with_errors)} unique tiles with errors in validation report")
    print()
    
    # Find intersection
    completed_with_errors = {}
    for tile_id in completed_tiles:
        if tile_id in tiles_with_errors:
            completed_with_errors[tile_id] = tiles_with_errors[tile_id]
    
    # Count total error instances (some tiles may have errors in multiple years)
    total_error_instances = sum(len(years) for years in completed_with_errors.values())
    
    # Print summary
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Completed tiles: {len(completed_tiles)}")
    print(f"Unique tiles with errors: {len(tiles_with_errors)}")
    print(f"Completed tiles WITH errors: {len(completed_with_errors)}")
    print(f"Total error instances (tile,year pairs): {total_error_instances}")
    print(f"Percentage of completed tiles with errors: {len(completed_with_errors)/len(completed_tiles)*100:.2f}%")
    print()
    
    # Write detailed report
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPLETED TILES WITH ERRORS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total completed tiles: {len(completed_tiles)}\n")
        f.write(f"Completed tiles with errors: {len(completed_with_errors)}\n")
        f.write(f"Total error instances (tile,year pairs): {total_error_instances}\n")
        f.write(f"Error rate: {len(completed_with_errors)/len(completed_tiles)*100:.2f}%\n")
        f.write("\n")
        
        # Group by number of years with errors
        tiles_by_year_count = {}
        for tile_id, years in completed_with_errors.items():
            year_count = len(years)
            if year_count not in tiles_by_year_count:
                tiles_by_year_count[year_count] = []
            tiles_by_year_count[year_count].append(tile_id)
        
        f.write("TILES GROUPED BY NUMBER OF YEARS WITH ERRORS:\n")
        f.write("-" * 80 + "\n")
        for year_count in sorted(tiles_by_year_count.keys(), reverse=True):
            tile_list = tiles_by_year_count[year_count]
            f.write(f"{year_count} year(s): {len(tile_list)} tiles\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("CSV FORMAT: tile_id,year1;year2;...\n")
        f.write("=" * 80 + "\n")
        for tile_id in sorted(completed_with_errors.keys()):
            years = ';'.join(sorted(completed_with_errors[tile_id]))
            f.write(f"{tile_id},{years}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SIMPLE CSV FORMAT: tile_id,year (one row per tile-year pair)\n")
        f.write("=" * 80 + "\n")
        for tile_id in sorted(completed_with_errors.keys()):
            for year in sorted(completed_with_errors[tile_id]):
                f.write(f"{tile_id},{year}\n")
    
    print(f"Detailed report written to: {OUTPUT_FILE}")
    print()
    
    # Print sample of problematic tiles
    if completed_with_errors:
        print("SAMPLE OF TILES WITH ERRORS (first 10):")
        print("-" * 80)
        for i, (tile_id, years) in enumerate(list(completed_with_errors.items())[:10]):
            years_str = ', '.join(sorted(years))
            print(f"{i+1}. {tile_id}")
            print(f"   Years with errors: {years_str}")
        
        if len(completed_with_errors) > 10:
            print(f"\n... and {len(completed_with_errors) - 10} more tiles with errors")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
