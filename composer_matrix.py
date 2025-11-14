"""
Create a square matrix showing binary classification accuracy
between each pair of the top 3 composers, with threshold graphs for each pair.
"""

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_preprocessed_data, get_composer_labels
from collections import Counter


def encode_labels(Y, composer_names):
    """Encode composer labels as binary: 0 for first composer, 1 for second."""
    labels = []
    for composer, piece in Y:
        if composer == composer_names[0]:
            labels.append(0)
        elif composer == composer_names[1]:
            labels.append(1)
        else:
            raise ValueError(f"Unexpected composer: {composer}")
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
    
    return best_threshold, thresholds


def evaluate_classifier(y_pred, y_true):
    """Evaluate classifier performance."""
    balanced_acc = calculate_balanced_accuracy(y_pred, y_true)
    accuracy = np.mean(y_pred == y_true)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'balanced_accuracy': balanced_acc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def binary_classify_pair(X, Y, composer1, composer2, test_size=0.3, random_seed=42):
    """Perform binary classification between two composers."""
    # Filter to only these two composers
    indices = [i for i, (composer, piece) in enumerate(Y) 
               if composer == composer1 or composer == composer2]
    
    if len(indices) < 20:
        return None
    
    X_pair = X[indices]
    Y_pair = [Y[i] for i in indices]
    y_pair = encode_labels(Y_pair, [composer1, composer2])
    
    # Split train/test
    n_samples = len(X_pair)
    n_train = int((1 - test_size) * n_samples)
    
    np.random.seed(random_seed)
    indices_shuffled = np.random.permutation(n_samples)
    train_indices = indices_shuffled[:n_train]
    test_indices = indices_shuffled[n_train:]
    
    X_train = X_pair[train_indices]
    X_test = X_pair[test_indices]
    y_train = y_pair[train_indices]
    y_test = y_pair[test_indices]
    
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
    
    # Find optimal threshold
    optimal_threshold, all_thresholds = find_optimal_threshold(y_train_pred_raw, y_train)
    
    # Classify test set
    y_test_pred = (y_test_pred_raw >= optimal_threshold).astype(int)
    results = evaluate_classifier(y_test_pred, y_test)
    
    return {
        'balanced_accuracy': results['balanced_accuracy'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'threshold': optimal_threshold,
        'all_thresholds': all_thresholds,
        'y_train_pred_raw': y_train_pred_raw,
        'y_train': y_train,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'composer1': composer1,
        'composer2': composer2
    }


def plot_threshold_graph(result, pair_num):
    """Plot threshold optimization graph for a composer pair."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate balanced accuracies for all thresholds
    balanced_accuracies = []
    for threshold in result['all_thresholds']:
        y_pred = (result['y_train_pred_raw'] >= threshold).astype(int)
        balanced_acc = calculate_balanced_accuracy(y_pred, result['y_train'])
        balanced_accuracies.append(balanced_acc)
    
    # Plot
    ax.plot(result['all_thresholds'], balanced_accuracies, linewidth=2, color='purple', label='Balanced Accuracy')
    ax.axvline(x=result['threshold'], color='green', linestyle='--', linewidth=2, 
               label=f'Optimal Threshold = {result["threshold"]:.4f}')
    ax.axhline(y=result['balanced_accuracy'], color='red', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Test Balanced Accuracy = {result["balanced_accuracy"]:.4f}')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title(f'{result["composer1"]} vs {result["composer2"]}\n'
                 f'Test Balanced Accuracy: {result["balanced_accuracy"]:.4f} ({result["balanced_accuracy"]*100:.2f}%)',
                 fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    filename = f'threshold_{result["composer1"]}_vs_{result["composer2"]}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def create_composer_matrix(X, Y, top_n=3):
    """Create a square matrix of binary classification accuracies."""
    composer_labels = get_composer_labels(Y)
    composer_counts = Counter(composer_labels)
    top_composers = [composer for composer, _ in composer_counts.most_common(top_n)]
    
    print(f"\nCreating {top_n}x{top_n} matrix for composers: {top_composers}")
    print("=" * 80)
    
    n = len(top_composers)
    accuracy_matrix = np.zeros((n, n))
    pair_results = []
    
    # Test each unique pair (only upper triangle)
    for i in range(n):
        for j in range(i + 1, n):
            composer1 = top_composers[i]
            composer2 = top_composers[j]
            
            print(f"\nPair {len(pair_results) + 1}: {composer1} vs {composer2}...")
            result = binary_classify_pair(X, Y, composer1, composer2)
            
            if result:
                accuracy_matrix[i, j] = result['balanced_accuracy']
                accuracy_matrix[j, i] = result['balanced_accuracy']  # Symmetric
                pair_results.append(result)
                print(f"  Balanced Accuracy: {result['balanced_accuracy']:.4f} ({result['balanced_accuracy']*100:.2f}%)")
                print(f"  Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1']:.4f}")
                
                # Plot threshold graph
                plot_threshold_graph(result, len(pair_results))
            else:
                print("  Insufficient data")
    
    # Diagonal is 1.0 (same composer)
    for i in range(n):
        accuracy_matrix[i, i] = 1.0
    
    return accuracy_matrix, top_composers, pair_results


def visualize_matrix(accuracy_matrix, composer_names):
    """Visualize the accuracy matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Balanced Accuracy', rotation=270, labelpad=20)
    
    ax.set_xticks(np.arange(len(composer_names)))
    ax.set_yticks(np.arange(len(composer_names)))
    ax.set_xticklabels(composer_names)
    ax.set_yticklabels(composer_names)
    
    # Add text annotations
    for i in range(len(composer_names)):
        for j in range(len(composer_names)):
            if np.isnan(accuracy_matrix[i, j]):
                text = ax.text(j, i, 'N/A', ha="center", va="center", color="black")
            else:
                text = ax.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                              ha="center", va="center", 
                              color="white" if accuracy_matrix[i, j] < 0.75 else "black",
                              fontweight='bold')
    
    ax.set_xlabel('Composer 2 (Predicted)', fontsize=12)
    ax.set_ylabel('Composer 1 (True)', fontsize=12)
    ax.set_title('Binary Classification Balanced Accuracy Matrix\n(Normalized for Class Imbalance)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('composer_matrix.png', dpi=150, bbox_inches='tight')
    print("\nMatrix visualization saved as 'composer_matrix.png'")
    plt.close()


def main():
    print("=" * 80)
    print("Composer Binary Classification Matrix")
    print("=" * 80)
    
    # Load data
    print("\nLoading preprocessed data...")
    X, Y, feature_names = load_preprocessed_data("data")
    print(f"Loaded {len(X)} segments with {len(feature_names)} features")
    
    # Create matrix
    accuracy_matrix, composer_names, pair_results = create_composer_matrix(X, Y, top_n=3)
    
    # Print matrix
    print("\n" + "=" * 80)
    print("BALANCED ACCURACY MATRIX")
    print("=" * 80)
    print(f"\n{'':15s}", end="")
    for name in composer_names:
        print(f"{name:15s}", end="")
    print()
    
    for i, name in enumerate(composer_names):
        print(f"{name:15s}", end="")
        for j in range(len(composer_names)):
            print(f"{accuracy_matrix[i, j]:.4f}        ", end="")
        print()
    
    # Visualize matrix
    print("\nGenerating matrix visualization...")
    visualize_matrix(accuracy_matrix, composer_names)
    
    print("\n" + "=" * 80)
    print("Complete! Generated files:")
    print("  - composer_matrix.png (accuracy matrix)")
    for result in pair_results:
        print(f"  - threshold_{result['composer1']}_vs_{result['composer2']}.png")
    print("=" * 80)
    
    return accuracy_matrix, composer_names, pair_results


if __name__ == "__main__":
    np.random.seed(42)
    accuracy_matrix, composer_names, pair_results = main()
