prefix = "/home/pringle/sentineldownloader/tessera"
input_filepath = "shapefiles/DISTRICT_BOUNDARY.shp"
output_filepath = "shapefiles/reprojected_india.shp"

#!/usr/bin/env python3
import os
import geopandas as gpd
from pyproj import CRS
import logging

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
target_epsg = 4326  # WGS84

# -----------------------------
# Paths
# -----------------------------
input_shp = os.path.join(prefix, input_filepath)
output_shp = os.path.join(prefix, output_filepath)

# -----------------------------
# Load shapefile
# -----------------------------
logger.info(f"Loading shapefile: {input_shp}")
gdf = gpd.read_file(input_shp)
logger.info(f"Shapefile CRS: {gdf.crs}")
logger.info(f"Number of features: {len(gdf)}")

# -----------------------------
# Reproject to EPSG:4326
# -----------------------------
logger.info(f"Reprojecting to EPSG:{target_epsg} (WGS84)...")
gdf_4326 = gdf.to_crs(CRS.from_epsg(target_epsg))
logger.info(f"Reprojection complete. New CRS: {gdf_4326.crs}")

# -----------------------------
# Save reprojected shapefile
# -----------------------------
os.makedirs(os.path.dirname(output_shp), exist_ok=True)
logger.info(f"Saving reprojected shapefile to: {output_shp}")
gdf_4326.to_file(output_shp)
logger.info("✅ Shapefile saved successfully")

