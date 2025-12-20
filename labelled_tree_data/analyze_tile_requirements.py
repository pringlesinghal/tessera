#!/usr/bin/env python3
"""
Analyze tile requirements for labeled tree dataset.

This script analyzes the mapped coordinate results to understand:
1. Which tiles contain labeled data
2. How many samples per tile
3. Species distribution across tiles
4. Coverage statistics
"""

import pandas as pd
from pathlib import Path
from collections import Counter

def analyze_dataset(mapped_csv_path: Path, dataset_name: str):
    """Analyze a single mapped dataset."""
    print(f"\n=== Analysis for {dataset_name} ===")
    
    df = pd.read_csv(mapped_csv_path)
    
    # Basic stats
    total_samples = len(df)
    successful_mappings = df['tile_id'].notna().sum()
    unique_tiles = df['tile_id'].nunique()
    unique_species = df['class'].nunique()
    
    print(f"Total samples: {total_samples}")
    print(f"Successful mappings: {successful_mappings}")
    print(f"Unique tiles: {unique_tiles}")
    print(f"Unique species: {unique_species}")
    
    # Tile distribution
    print(f"\nSamples per tile (top 10):")
    tile_counts = df['tile_id'].value_counts().head(10)
    for tile_id, count in tile_counts.items():
        print(f"  {tile_id}: {count} samples")
    
    # Species distribution
    print(f"\nSpecies distribution:")
    species_counts = df['class'].value_counts()
    for species, count in species_counts.items():
        print(f"  {species}: {count} samples")
    
    # Coverage statistics
    valid_coverage = df[df['coverage_ratio'].notna()]['coverage_ratio']
    print(f"\nCoverage statistics:")
    print(f"  Mean coverage: {valid_coverage.mean():.3f}")
    print(f"  Min coverage: {valid_coverage.min():.3f}")
    print(f"  Max coverage: {valid_coverage.max():.3f}")
    print(f"  Full coverage tiles (1.0): {(valid_coverage == 1.0).sum()}")
    
    return {
        'dataset': dataset_name,
        'total_samples': total_samples,
        'unique_tiles': unique_tiles,
        'unique_species': unique_species,
        'tile_counts': tile_counts,
        'species_counts': species_counts,
        'df': df
    }

def analyze_all_datasets():
    """Analyze all mapped datasets."""
    base_dir = Path("/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/labelled_tree_data")
    
    datasets = {
        'train': base_dir / 'train_mapped.csv',
        'val': base_dir / 'val_mapped.csv', 
        'test': base_dir / 'test_mapped.csv'
    }
    
    results = {}
    all_tiles = set()
    all_species = set()
    
    for name, path in datasets.items():
        if path.exists():
            result = analyze_dataset(path, name)
            results[name] = result
            all_tiles.update(result['df']['tile_id'].dropna())
            all_species.update(result['df']['class'])
        else:
            print(f"Warning: {path} not found")
    
    # Combined analysis
    print(f"\n=== Combined Analysis ===")
    print(f"Total unique tiles across all datasets: {len(all_tiles)}")
    print(f"Total unique species: {len(all_species)}")
    
    # Find tiles that appear in multiple datasets
    train_tiles = set(results['train']['df']['tile_id'].dropna())
    val_tiles = set(results['val']['df']['tile_id'].dropna())
    test_tiles = set(results['test']['df']['tile_id'].dropna())
    
    train_val_overlap = train_tiles & val_tiles
    train_test_overlap = train_tiles & test_tiles
    val_test_overlap = val_tiles & test_tiles
    all_overlap = train_tiles & val_tiles & test_tiles
    
    print(f"\nTile overlap between datasets:")
    print(f"  Train-Val overlap: {len(train_val_overlap)} tiles")
    print(f"  Train-Test overlap: {len(train_test_overlap)} tiles")
    print(f"  Val-Test overlap: {len(val_test_overlap)} tiles")
    print(f"  All three overlap: {len(all_overlap)} tiles")
    
    # Generate tile priority list
    tile_priority = Counter()
    for name, result in results.items():
        for tile_id in result['df']['tile_id'].dropna():
            tile_priority[tile_id] += 1
    
    print(f"\nTile priority (appears in multiple datasets):")
    for tile_id, count in tile_priority.most_common(20):
        datasets_str = []
        if tile_id in train_tiles:
            datasets_str.append("train")
        if tile_id in val_tiles:
            datasets_str.append("val")
        if tile_id in test_tiles:
            datasets_str.append("test")
        print(f"  {tile_id}: {count} datasets ({', '.join(datasets_str)})")
    
    # Save required tiles list
    required_tiles_path = base_dir / "required_tiles.txt"
    with open(required_tiles_path, 'w') as f:
        for tile_id in sorted(all_tiles):
            f.write(f"{tile_id}\n")
    print(f"\nSaved {len(all_tiles)} required tiles to: {required_tiles_path}")
    
    # Save high-priority tiles (appear in multiple datasets)
    high_priority_tiles = [tile for tile, count in tile_priority.items() if count > 1]
    priority_tiles_path = base_dir / "priority_tiles.txt"
    with open(priority_tiles_path, 'w') as f:
        for tile_id in sorted(high_priority_tiles):
            f.write(f"{tile_id}\n")
    print(f"Saved {len(high_priority_tiles)} high-priority tiles to: {priority_tiles_path}")
    
    return results, all_tiles, all_species

if __name__ == "__main__":
    results, all_tiles, all_species = analyze_all_datasets()
    
    print(f"\n=== Species Summary ===")
    print("All species in dataset:")
    for species in sorted(all_species):
        print(f"  {species}")