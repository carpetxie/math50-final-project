"""
Quick script to explore the preprocessed data interactively.
"""

import numpy as np
from load_data import load_preprocessed_data, get_composer_labels

# Load the data
X, Y, feature_names = load_preprocessed_data()

# Print basic info
print(f"X shape: {X.shape}")
print(f"Number of segments: {len(Y)}")
print(f"\nFeature names: {feature_names}")

# Show first few examples
print("\n" + "="*80)
print("First 5 examples:")
print("="*80)
for i in range(min(5, len(Y))):
    composer, piece = Y[i]
    print(f"\nSegment {i+1}:")
    print(f"  Composer: {composer}")
    print(f"  Piece: {piece}")
    print(f"  Features:")
    for j, name in enumerate(feature_names):
        print(f"    {name:20s}: {X[i, j]:8.2f}")

# Show feature statistics
print("\n" + "="*80)
print("Feature Statistics:")
print("="*80)
for i, name in enumerate(feature_names):
    print(f"{name:20s}: mean={X[:, i].mean():8.2f}, std={X[:, i].std():8.2f}")

# Show composer distribution
print("\n" + "="*80)
print("Composer Distribution:")
print("="*80)
composer_labels = get_composer_labels(Y)
unique_composers, counts = np.unique(composer_labels, return_counts=True)
for composer, count in zip(unique_composers, counts):
    print(f"  {composer:20s}: {count:4d} segments ({100*count/len(Y):5.1f}%)")

# Access specific data
print("\n" + "="*80)
print("Example: Get all Bach segments")
print("="*80)
bach_indices = [i for i, (composer, piece) in enumerate(Y) if composer == 'bach']
print(f"Found {len(bach_indices)} Bach segments")
if len(bach_indices) > 0:
    bach_X = X[bach_indices]
    print(f"Bach feature means: {bach_X.mean(axis=0)}")

