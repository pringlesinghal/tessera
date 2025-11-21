import ee, os, re
ee.Initialize(project='povertyparking')

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
TILE_LIST_FILE = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/tiles_utm/completed_tiles.txt"
LOCAL_TILE_DIR = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/tiles_utm"
EXPORT_BUCKET = "sidd_rajasthan"
EXPORT_PREFIX = "psinghal/farmtree_tiles"

# -------------------------------------------------------------------
# 1. LOAD TILE IDS FROM FILE
# -------------------------------------------------------------------
with open(TILE_LIST_FILE, "r") as f:
    tile_ids = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(tile_ids)} completed tiles")

# Utility: parse tile ID
TILE_RE = re.compile(r"zone(\d+)_r(\d+)_c(\d+)")

def parse_tile_id(tile_str):
    m = TILE_RE.match(tile_str)
    if not m:
        raise ValueError(f"Invalid tile id format: {tile_str}")
    zone = int(m.group(1))
    r = int(m.group(2))
    c = int(m.group(3))
    epsg = zone
    return zone, r, c, epsg

# -------------------------------------------------------------------
# 2. LOAD TREE/CROPLAND MASK
# -------------------------------------------------------------------
# Tree cover merged mosaic (already prepared)
treeCoverCollections = [
    ee.ImageCollection(f'projects/ee-rscph-{pid}/assets/tree/global')
    for pid in [
        2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        17,18,19,20,21,22,23,24,25,26,27,28,
        29,30,31,47,48,49
    ]
]

# Mosaic all tree cover
tree_cover = treeCoverCollections[0]
for c in treeCoverCollections[1:]:
    tree_cover = tree_cover.merge(c)
tree_cover = tree_cover.mosaic()

# Cropland mask
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
cropland_mask = worldcover.eq(40)

# -------------------------------------------------------------------
# 3. EXPORT MASKS ONLY FOR TILES PRESENT LOCALLY
# -------------------------------------------------------------------
def export_masks_for_tile(tile_id):
    # Check local file exists
    tiff_path = os.path.join(LOCAL_TILE_DIR, f"tile_{tile_id}.tif")
    if not os.path.exists(tiff_path):
        print(f"Skipping {tile_id}: Local file missing")
        return

    zone, r, c, epsg = parse_tile_id(tile_id)

    # Load tile geometry from the TIFF
    import rasterio
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
    xmin, ymin, xmax, ymax = bounds

    tile_region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax],
                                        proj=f"EPSG:{epsg}",
                                        geodesic=False)

    # Reproject tree cover into tile CRS at 10m
    tree_res_10m = tree_cover.reproject(
        crs=f"EPSG:{epsg}",
        scale=10
    )

    # Compute 10m averaged tree cover then mask at >10%
    mean_tree = tree_res_10m.reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=True,
        maxPixels=1024
    ).reproject(crs=f"EPSG:{epsg}", scale=10)

    farm_tree_mask = mean_tree.gt(0.1) \
        .And(cropland_mask.reproject(crs=f"EPSG:{epsg}", scale=10)) \
        .selfMask() \
        .uint8()

    # Export
    task = ee.batch.Export.image.toCloudStorage(
        image=farm_tree_mask,
        description=f"farmtree_{tile_id}",
        bucket=EXPORT_BUCKET,
        fileNamePrefix=f"{EXPORT_PREFIX}/{tile_id}",
        region=tile_region,
        scale=10,
        crs=f"EPSG:{epsg}",
        maxPixels=1e12
    )
    task.start()

    print(f"Submitted tile: {tile_id}")

# -------------------------------------------------------------------
# 4. RUN EXPORTS
# -------------------------------------------------------------------
for tile_id in tile_ids:
    export_masks_for_tile(tile_id)

print("All valid tile exports submitted.")
