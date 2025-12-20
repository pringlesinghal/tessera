#!/usr/bin/env python3
"""
Coordinate Mapper for Tree Species Validation Dataset

Converts lat/long coordinates to tile IDs and pixel coordinates within the 
existing tessera tile system.

Key functionality:
1. Convert WGS84 coordinates to UTM 
2. Map UTM coordinates to existing tile structure
3. Calculate pixel coordinates within tiles (10m resolution)
4. Handle edge cases and validate coordinates
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from pyproj import Transformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CoordinateMapping:
    """Result of coordinate mapping operation"""
    latitude: float
    longitude: float
    tile_id: str
    parent_tile_id: str
    zone: int
    epsg: int
    utm_easting: float
    utm_northing: float
    pixel_row: int
    pixel_col: int
    within_tile_bounds: bool
    coverage_ratio: float

class TileInfo(NamedTuple):
    """Tile information from manifest"""
    tile_id: str
    parent_tile_id: str
    zone: int
    epsg: int
    min_e: float
    min_n: float
    max_e: float
    max_n: float
    coverage_ratio: float

class CoordinateMapper:
    """
    Maps lat/long coordinates to tessera tile system coordinates.
    
    Uses the existing subtile manifest to identify tiles and calculate
    pixel coordinates within tiles at 10m resolution.
    """
    
    def __init__(self, manifest_path: Path):
        """
        Initialize mapper with subtile manifest.
        
        Args:
            manifest_path: Path to subtile_manifest.csv
        """
        self.manifest_path = Path(manifest_path)
        self.pixel_size = 10.0  # 10m resolution
        self.tile_size = 20000  # 20km tiles
        
        # Load and process manifest
        self._load_manifest()
        self._build_spatial_index()
    
    def _load_manifest(self):
        """Load the subtile manifest CSV file."""
        logger.info(f"Loading manifest from {self.manifest_path}")
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        self.manifest_df = pd.read_csv(self.manifest_path)
        logger.info(f"Loaded {len(self.manifest_df)} tiles from manifest")
        
        # Validate required columns
        required_cols = ['tile_id', 'parent_tile_id', 'zone', 'epsg', 
                        'min_e', 'min_n', 'max_e', 'max_n', 'coverage_ratio']
        missing_cols = set(required_cols) - set(self.manifest_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in manifest: {missing_cols}")
    
    def _build_spatial_index(self):
        """Build spatial index grouped by EPSG for efficient lookup."""
        self.tiles_by_epsg = {}
        
        for _, row in self.manifest_df.iterrows():
            epsg = row['epsg']
            if epsg not in self.tiles_by_epsg:
                self.tiles_by_epsg[epsg] = []
            
            tile_info = TileInfo(
                tile_id=row['tile_id'],
                parent_tile_id=row['parent_tile_id'],
                zone=row['zone'], 
                epsg=epsg,
                min_e=row['min_e'],
                min_n=row['min_n'],
                max_e=row['max_e'],
                max_n=row['max_n'],
                coverage_ratio=row['coverage_ratio']
            )
            self.tiles_by_epsg[epsg].append(tile_info)
        
        logger.info(f"Built spatial index for {len(self.tiles_by_epsg)} EPSG codes")
    
    def _determine_utm_zone(self, longitude: float) -> Tuple[int, bool]:
        """
        Determine UTM zone and hemisphere from longitude.
        
        Args:
            longitude: Longitude in decimal degrees
            
        Returns:
            Tuple of (zone_number, is_northern)
        """
        zone = int(math.floor((longitude + 180) / 6) + 1)
        zone = max(1, min(60, zone))  # Clamp to valid range
        
        # For India, we're always in northern hemisphere
        return zone, True
    
    def _get_utm_epsg(self, zone: int, northern: bool) -> int:
        """Get EPSG code for UTM zone."""
        return (32600 if northern else 32700) + zone
    
    def _convert_to_utm(self, latitude: float, longitude: float) -> Tuple[int, float, float]:
        """
        Convert lat/long to UTM coordinates.
        
        Args:
            latitude: Latitude in decimal degrees  
            longitude: Longitude in decimal degrees
            
        Returns:
            Tuple of (epsg_code, easting, northing)
        """
        zone, northern = self._determine_utm_zone(longitude)
        epsg = self._get_utm_epsg(zone, northern)
        
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        easting, northing = transformer.transform(longitude, latitude)
        
        return epsg, easting, northing
    
    def _find_containing_tile(self, epsg: int, easting: float, northing: float) -> Optional[TileInfo]:
        """
        Find tile that contains the given UTM coordinates.
        
        Args:
            epsg: UTM EPSG code
            easting: UTM easting coordinate
            northing: UTM northing coordinate
            
        Returns:
            TileInfo if found, None otherwise
        """
        if epsg not in self.tiles_by_epsg:
            return None
        
        for tile in self.tiles_by_epsg[epsg]:
            if (tile.min_e <= easting < tile.max_e and 
                tile.min_n <= northing < tile.max_n):
                return tile
        
        return None
    
    def _calculate_pixel_coordinates(self, tile: TileInfo, easting: float, northing: float) -> Tuple[int, int]:
        """
        Calculate pixel row/col within tile.
        
        Args:
            tile: TileInfo object
            easting: UTM easting coordinate
            northing: UTM northing coordinate
            
        Returns:
            Tuple of (pixel_row, pixel_col)
        """
        # Calculate relative position within tile
        rel_e = easting - tile.min_e
        rel_n = northing - tile.min_n
        
        # Convert to pixel coordinates
        # Note: Row increases from north to south, so we flip northing
        pixel_col = int(rel_e / self.pixel_size)
        pixel_row = int((self.tile_size - rel_n) / self.pixel_size)
        
        # Clamp to tile bounds (should be 0-1999 for 20km tiles at 10m resolution)
        max_pixel = int(self.tile_size / self.pixel_size) - 1
        pixel_row = max(0, min(max_pixel, pixel_row))
        pixel_col = max(0, min(max_pixel, pixel_col))
        
        return pixel_row, pixel_col
    
    def map_coordinate(self, latitude: float, longitude: float) -> Optional[CoordinateMapping]:
        """
        Map a single coordinate to tile system.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            CoordinateMapping object if successful, None if coordinate can't be mapped
        """
        try:
            # Convert to UTM
            epsg, easting, northing = self._convert_to_utm(latitude, longitude)
            
            # Find containing tile
            tile = self._find_containing_tile(epsg, easting, northing)
            if tile is None:
                logger.warning(f"No tile found for coordinates ({latitude}, {longitude})")
                return None
            
            # Calculate pixel coordinates
            pixel_row, pixel_col = self._calculate_pixel_coordinates(tile, easting, northing)
            
            return CoordinateMapping(
                latitude=latitude,
                longitude=longitude,
                tile_id=tile.tile_id,
                parent_tile_id=tile.parent_tile_id,
                zone=tile.zone,
                epsg=epsg,
                utm_easting=easting,
                utm_northing=northing,
                pixel_row=pixel_row,
                pixel_col=pixel_col,
                within_tile_bounds=True,
                coverage_ratio=tile.coverage_ratio
            )
            
        except Exception as e:
            logger.error(f"Error mapping coordinate ({latitude}, {longitude}): {e}")
            return None
    
    def map_coordinates_batch(self, coordinates: List[Tuple[float, float]]) -> List[Optional[CoordinateMapping]]:
        """
        Map multiple coordinates efficiently.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            
        Returns:
            List of CoordinateMapping objects (None for failed mappings)
        """
        results = []
        
        for lat, lon in coordinates:
            result = self.map_coordinate(lat, lon)
            results.append(result)
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Successfully mapped {successful}/{len(coordinates)} coordinates")
        
        return results
    
    def map_from_csv(self, csv_path: Path, lat_col: str = 'latitude', 
                     lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Map coordinates from a CSV file.
        
        Args:
            csv_path: Path to CSV file with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            DataFrame with original data plus mapping results
        """
        logger.info(f"Processing coordinates from {csv_path}")
        
        df = pd.read_csv(csv_path)
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Required columns {lat_col}, {lon_col} not found")
        
        # Extract coordinates
        coordinates = list(zip(df[lat_col], df[lon_col]))
        
        # Map coordinates
        mappings = self.map_coordinates_batch(coordinates)
        
        # Add mapping results to dataframe
        mapping_data = []
        for mapping in mappings:
            if mapping is not None:
                mapping_data.append({
                    'tile_id': mapping.tile_id,
                    'parent_tile_id': mapping.parent_tile_id,
                    'zone': mapping.zone,
                    'epsg': mapping.epsg,
                    'utm_easting': mapping.utm_easting,
                    'utm_northing': mapping.utm_northing,
                    'pixel_row': mapping.pixel_row,
                    'pixel_col': mapping.pixel_col,
                    'within_tile_bounds': mapping.within_tile_bounds,
                    'coverage_ratio': mapping.coverage_ratio
                })
            else:
                mapping_data.append({
                    'tile_id': None,
                    'parent_tile_id': None,
                    'zone': None,
                    'epsg': None,
                    'utm_easting': None,
                    'utm_northing': None,
                    'pixel_row': None,
                    'pixel_col': None,
                    'within_tile_bounds': False,
                    'coverage_ratio': None
                })
        
        # Combine with original data
        mapping_df = pd.DataFrame(mapping_data)
        result_df = pd.concat([df, mapping_df], axis=1)
        
        return result_df
    
    def get_required_tiles(self, coordinates: List[Tuple[float, float]]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get list of unique tiles required for given coordinates.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            
        Returns:
            Dictionary mapping tile_id to list of coordinates in that tile
        """
        tile_coords = {}
        
        mappings = self.map_coordinates_batch(coordinates)
        
        for coord, mapping in zip(coordinates, mappings):
            if mapping is not None:
                tile_id = mapping.tile_id
                if tile_id not in tile_coords:
                    tile_coords[tile_id] = []
                tile_coords[tile_id].append(coord)
        
        logger.info(f"Found {len(tile_coords)} unique tiles for {len(coordinates)} coordinates")
        
        return tile_coords


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test coordinate mapping")
    parser.add_argument("--manifest", 
                       default="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/shapefiles/india_tiles/mgrs_tiles/subtile_manifest.csv",
                       help="Path to subtile manifest")
    parser.add_argument("--csv", 
                       default="/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/labelled_tree_data/train.csv",
                       help="Path to CSV with coordinates")
    parser.add_argument("--output", 
                       help="Output path for mapped results")
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = CoordinateMapper(Path(args.manifest))
    
    # Test single coordinate mapping
    print("\n=== Testing Single Coordinate ===")
    test_lat, test_lon = 25.78219989847472, 73.32243100156258  # First coordinate from train.csv
    result = mapper.map_coordinate(test_lat, test_lon)
    
    if result:
        print(f"Coordinate: ({test_lat}, {test_lon})")
        print(f"Tile ID: {result.tile_id}")
        print(f"Parent Tile: {result.parent_tile_id}")
        print(f"UTM: {result.utm_easting:.1f}, {result.utm_northing:.1f}")
        print(f"Pixel: row={result.pixel_row}, col={result.pixel_col}")
        print(f"Coverage: {result.coverage_ratio:.3f}")
    else:
        print("Failed to map coordinate")
    
    # Test CSV mapping
    if Path(args.csv).exists():
        print(f"\n=== Processing CSV: {args.csv} ===")
        mapped_df = mapper.map_from_csv(Path(args.csv))
        
        print(f"Mapped {len(mapped_df)} rows")
        print(f"Successful mappings: {mapped_df['tile_id'].notna().sum()}")
        print(f"Unique tiles: {mapped_df['tile_id'].nunique()}")
        
        if args.output:
            mapped_df.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")
        
        # Show sample results
        print("\nSample mappings:")
        sample = mapped_df[mapped_df['tile_id'].notna()].head(3)
        for _, row in sample.iterrows():
            print(f"  {row['class']}: {row['tile_id']} -> pixel({row['pixel_row']}, {row['pixel_col']})")