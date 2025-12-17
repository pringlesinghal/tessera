#!/bin/bash

# List of tile IDs
tiles=(
  "zone32642_r19_c14"
  "zone32642_r18_c4"
  "zone32642_r18_c9"
  "zone32642_r13_c15"
  "zone32642_r10_c10"
)

# Base GCS bucket path
base_path="gs://sidd_rajasthan/psinghal/time_series_utm"

# Local output directory
out_dir="./downloaded_tiles"
mkdir -p "$out_dir"

# Loop over tiles first, years inside
for tile in "${tiles[@]}"; do
  echo "=== TILE $tile ==="

  for year in {2016..2024}; do
    echo "Downloading $tile for $year ..."

    remote="${base_path}/${year}/${tile}/${tile}/data_processed"
    local="${out_dir}/${tile}/${year}"

    mkdir -p "$local"

    # Download the entire folder (serial)
    gsutil -m cp -r "$remote" "$local"

    echo "Finished $tile for $year"
    echo
  done
done

echo "All downloads completed."
