"""
Feature ablation study to determine which features are most important
for classifying each composer.

For each composer, performs binary classification (this composer vs all others)
and ablates each feature to measure its impact on accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_preprocessed_data, get_composer_labels
from collections import Counter


def encode_binary_labels(Y, target_composer):
    """Encode labels as 1 for target composer, 0 for all others."""
    labels = []
    for composer, piece in Y:
        labels.append(1 if composer == target_composer else 0)
    return np.array(labels).reshape(-1, 1)


def add_bias_column(X):
    """Add a column of ones for the bias/intercept term."""
    n_samples = X.shape[0]
    bias = np.ones((n_samples, 1))
    return np.hstack([bias, X])


def standardize_features(X_train, X_test):
    """Standardize features to have mean 0 and std 1."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled


def solve_least_squares(A, b):
    """Solve Ax = b using least squares: x = (A^T A)^(-1) A^T b"""
    ATA = A.T @ A
    x = np.linalg.solve(ATA, A.T @ b)
    return x


def calculate_balanced_accuracy(y_pred, y_true):
    """Calculate balanced accuracy (accounts for class imbalance)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    balanced_acc = (sensitivity + specificity) / 2
    return balanced_acc


def find_optimal_threshold(y_pred_raw, y_true):
    """Find the optimal threshold that maximizes balanced accuracy."""
    min_pred = np.min(y_pred_raw)
    max_pred = np.max(y_pred_raw)
    thresholds = np.linspace(min_pred, max_pred, 100)
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_raw >= threshold).astype(int)
        balanced_acc = calculate_balanced_accuracy(y_pred, y_true)
        if balanced_acc > best_accuracy:
            best_accuracy = balanced_acc
            best_threshold = threshold
    
    return best_threshold


def classify_with_features(X, y, feature_indices, test_size=0.3, random_seed=42):
    """
    Perform binary classification using only specified features.
    
    Args:
        X: Full feature matrix
        y: Binary labels
        feature_indices: List of feature indices to use
        test_size: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Test accuracy
    """
    # Select only specified features
    X_subset = X[:, feature_indices]
    
    # Split train/test
    n_samples = len(X_subset)
    n_train = int((1 - test_size) * n_samples)
    
    np.random.seed(random_seed)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X_subset[train_indices]
    X_test = X_subset[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Standardize
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    
    # Add bias column
    A_train = add_bias_column(X_train_scaled)
    A_test = add_bias_column(X_test_scaled)
    
    # Solve least squares
    x = solve_least_squares(A_train, y_train)
    
    # Predict
    y_train_pred_raw = A_train @ x
    y_test_pred_raw = A_test @ x
    
    # Find optimal threshold on training set
    optimal_threshold = find_optimal_threshold(y_train_pred_raw, y_train)
    
    # Classify test set
    y_test_pred = (y_test_pred_raw >= optimal_threshold).astype(int)
    
    # Calculate balanced accuracy
    balanced_acc = calculate_balanced_accuracy(y_test_pred, y_test)
    
    return balanced_acc


def ablation_study_for_composer(X, Y, target_composer, feature_names):
    """
    Perform ablation study for one composer.
    
    Args:
        X: Feature matrix
        Y: List of (composer, piece) tuples
        target_composer: Composer to classify (vs all others)
        feature_names: List of feature names
        
    Returns:
        Dictionary with baseline accuracy and accuracy changes per feature
    """
    print(f"\nAblation study for {target_composer}...")
    print("-" * 60)
    
    # Encode as binary classification
    y = encode_binary_labels(Y, target_composer)
    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)
    print(f"  Positive samples ({target_composer}): {n_positive}")
    print(f"  Negative samples (others): {n_negative}")
    
    # Baseline: all features
    all_features = list(range(X.shape[1]))
    baseline_balanced_acc = classify_with_features(X, y, all_features)
    print(f"  Baseline balanced accuracy (all features): {baseline_balanced_acc:.4f} ({baseline_balanced_acc*100:.2f}%)")
    
    # Ablate each feature
    ablation_results = {}
    print(f"\n  Ablating each feature:")
    
    for i, feature_name in enumerate(feature_names):
        # Remove feature i
        features_without_i = [j for j in range(X.shape[1]) if j != i]
        
        # Train without this feature
        ablated_balanced_acc = classify_with_features(X, y, features_without_i)
        
        # Calculate change
        accuracy_change = baseline_balanced_acc - ablated_balanced_acc
        
        ablation_results[feature_name] = {
            'ablated_balanced_accuracy': ablated_balanced_acc,
            'accuracy_change': accuracy_change
        }
        
        change_str = f"{accuracy_change:+.4f}" if accuracy_change >= 0 else f"{accuracy_change:.4f}"
        print(f"    {feature_name:20s}: {ablated_balanced_acc:.4f} (change: {change_str})")
    
    return {
        'baseline_balanced_accuracy': baseline_balanced_acc,
        'ablation_results': ablation_results
    }


def create_ablation_heatmap(results, composer_names, feature_names):
    """
    Create a heatmap showing accuracy changes from feature ablation.
    
    Args:
        results: Dictionary of results for each composer
        composer_names: List of composer names
        feature_names: List of feature names
    """
    n_features = len(feature_names)
    n_composers = len(composer_names)
    
    # Build matrix: rows = features, columns = composers
    heatmap_data = np.zeros((n_features, n_composers))
    
    for j, composer in enumerate(composer_names):
        for i, feature in enumerate(feature_names):
            heatmap_data[i, j] = results[composer]['ablation_results'][feature]['accuracy_change']
    
    # Update colorbar range based on data
    vmin = np.min(heatmap_data) - 0.01
    vmax = np.max(heatmap_data) + 0.01
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Use diverging colormap (red = important, blue = harmful)
    # Center colormap at 0 for better visualization
    vmax_abs = max(abs(vmin), abs(vmax))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-vmax_abs, vmax=vmax_abs)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Balanced Accuracy Change (Baseline - Ablated)', rotation=270, labelpad=25)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_composers))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(composer_names)
    ax.set_yticklabels(feature_names)
    
    # Add text annotations
    for i in range(n_features):
        for j in range(n_composers):
            value = heatmap_data[i, j]
            text_color = "white" if abs(value) > 0.08 else "black"
            text = ax.text(j, i, f'{value:+.3f}',
                          ha="center", va="center",
                          color=text_color,
                          fontsize=10)
    
    ax.set_xlabel('Composer', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance via Ablation (Balanced Accuracy)\n(Positive = Important, Negative = Harmful, Normalized for Class Imbalance)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('feature_ablation_heatmap.png', dpi=150, bbox_inches='tight')
    print("\nHeatmap saved as 'feature_ablation_heatmap.png'")
    plt.show()


def print_summary(results, composer_names):
    """Print summary of most important features per composer."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)
    
    for composer in composer_names:
        print(f"\n{composer.upper()}:")
        print(f"  Baseline Balanced Accuracy: {results[composer]['baseline_balanced_accuracy']:.4f} "
              f"({results[composer]['baseline_balanced_accuracy']*100:.2f}%)")
        
        # Sort features by importance
        ablation_results = results[composer]['ablation_results']
        sorted_features = sorted(ablation_results.items(), 
                                key=lambda x: x[1]['accuracy_change'], 
                                reverse=True)
        
        print(f"\n  Top 5 Most Important Features:")
        for i, (feature, data) in enumerate(sorted_features[:5]):
            print(f"    {i+1}. {feature:20s}: {data['accuracy_change']:+.4f}")
        
        print(f"\n  Least Important Features:")
        for i, (feature, data) in enumerate(sorted_features[-3:]):
            print(f"    {feature:20s}: {data['accuracy_change']:+.4f}")


def main():
    print("=" * 80)
    print("Feature Ablation Study")
    print("=" * 80)
    
    # Load data
    print("\nLoading preprocessed data...")
    X, Y, feature_names = load_preprocessed_data("data")
    print(f"Loaded {len(X)} segments with {len(feature_names)} features")
    
    # Get top 3 composers
    composer_labels = get_composer_labels(Y)
    composer_counts = Counter(composer_labels)
    top_composers = [composer for composer, _ in composer_counts.most_common(3)]
    
    print(f"\nPerforming ablation study for: {top_composers}")
    
    # Run ablation study for each composer
    results = {}
    for composer in top_composers:
        results[composer] = ablation_study_for_composer(X, Y, composer, feature_names)
    
    # Create heatmap
    print("\nGenerating heatmap...")
    create_ablation_heatmap(results, top_composers, feature_names)
    
    # Print summary
    print_summary(results, top_composers)
    
    print("\n" + "=" * 80)
    print("Ablation study complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    results = main()

