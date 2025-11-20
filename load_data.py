"""
Utility script to load and inspect preprocessed MIDI data.
"""

import numpy as np
import pickle
from pathlib import Path


def load_preprocessed_data(data_dir="data"):
    """
    Load preprocessed X and Y data.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (X, Y, feature_names) where:
        - X is a numpy array of shape (n_segments, 9)
        - Y is a list of (composer, piece) tuples
        - feature_names is a list of feature names
    """
    X = np.load(f"{data_dir}/X_features.npy")
    
    with open(f"{data_dir}/Y_labels.pkl", "rb") as f:
        Y = pickle.load(f)
    
    with open(f"{data_dir}/feature_names.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return X, Y, feature_names


def print_data_summary(X, Y, feature_names):
    """Print a summary of the loaded data."""
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    
    print(f"\nX (Features):")
    print(f"  Shape: {X.shape}")
    print(f"  Data type: {X.dtype}")
    
    print(f"\nY (Labels):")
    print(f"  Total segments: {len(Y)}")
    print(f"  Unique composers: {len(set(y[0] for y in Y))}")
    print(f"  Unique pieces: {len(set(y for y in Y))}")
    
    print(f"\nFeatures ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    print("\n" + "-" * 80)
    print("Feature Statistics:")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"{name:20s}: mean={X[:, i].mean():8.2f}, std={X[:, i].std():8.2f}, "
              f"min={X[:, i].min():8.2f}, max={X[:, i].max():8.2f}")
    
    print("\n" + "-" * 80)
    print("Composer Distribution:")
    print("-" * 80)
    composer_counts = {}
    for composer, piece in Y:
        composer_counts[composer] = composer_counts.get(composer, 0) + 1
    
    for composer, count in sorted(composer_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(Y)
        print(f"  {composer:20s}: {count:4d} segments ({percentage:5.1f}%)")
    
    print("\n" + "-" * 80)
    print("Sample Data (first 5 segments):")
    print("-" * 80)
    for i in range(min(5, len(Y))):
        composer, piece = Y[i]
        print(f"\nSegment {i+1}:")
        print(f"  Composer: {composer}")
        print(f"  Piece: {piece}")
        print(f"  Features: {X[i]}")


def get_composer_labels(Y):
    """
    Extract just composer names for classification.
    
    Args:
        Y: List of (composer, piece) tuples
        
    Returns:
        List of composer names
    """
    return [composer for composer, piece in Y]


def filter_by_composer(X, Y, composers):
    """
    Filter dat to only include specified composers.
    
    Args:
        X: Feature matrix
        Y: List of (composer, piece) tuples
        composers: List of composer names to include
        
    Returns:
        Tuple of (X_filtered, Y_filtered)
    """
    indices = [i for i, (composer, piece) in enumerate(Y) if composer in composers]
    X_filtered = X[indices]
    Y_filtered = [Y[i] for i in indices]
    return X_filtered, Y_filtered


if __name__ == "__main__":
    # Load the data
    print("Loading preprocessed data...")
    X, Y, feature_names = load_preprocessed_data()
    
    # Print summary
    print_data_summary(X, Y, feature_names)
    
    # Example: Get just composer labels for classification
    composer_labels = get_composer_labels(Y)
    print("\n" + "=" * 80)
    print(f"Extracted {len(composer_labels)} composer labels for classification")
    print(f"Unique composers: {sorted(set(composer_labels))}")

