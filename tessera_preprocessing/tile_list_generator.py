import os
from pathlib import Path

# Directory containing your tiles
TILE_DIR = Path("/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/tiles_utm")

# Output file for the tile list
OUTPUT_FILE = TILE_DIR / "tile_list.txt"

# Find all .tif files matching the pattern
tile_files = sorted(TILE_DIR.glob("tile_*.tif"))

# Extract tile IDs (remove .tif and leading "tile_")
tile_ids = [f.stem.replace("tile_", "") for f in tile_files]

# Write to output file
with open(OUTPUT_FILE, "w") as f:
    for tile_id in tile_ids:
        f.write(f"{tile_id}\n")

print(f"Found {len(tile_ids)} tiles. Tile list saved to {OUTPUT_FILE}")