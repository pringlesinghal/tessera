#!/usr/bin/env python3
import sys
from pathlib import Path

import rasterio


def inspect_raster(tif_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        print(f"File: {tif_path}")
        print(f"CRS: {src.crs}")
        print(f"Dimensions (width x height): {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"Transform:\n{src.transform}")
        print(f"Dtype(s): {[src.dtypes[i] for i in range(src.count)]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_tif.py /path/to/file.tif")
        sys.exit(1)

    inspect_raster(Path(sys.argv[1]).resolve())

