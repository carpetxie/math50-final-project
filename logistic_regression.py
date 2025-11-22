"""
Simple logistic regression for binary classification between pairs of composers.
Creates a 3x3 accuracy matrix similar to composer_matrix.py but using logistic regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_preprocessed_data, get_composer_labels
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
    return np.array(labels)


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


def binary_classify_pair_logistic(X, Y, composer1, composer2, test_size=0.3, random_seed=42):
    """Perform binary logistic regression between two composers."""
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Get probabilities for visualization
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    results = evaluate_classifier(y_test_pred, y_test)
    
    return {
        'balanced_accuracy': results['balanced_accuracy'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'y_train_proba': y_train_proba,
        'y_train': y_train,
        'y_test_proba': y_test_proba,
        'y_test': y_test,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'composer1': composer1,
        'composer2': composer2,
        'model': model
    }


def plot_probability_distribution(result, pair_num):
    """Plot probability distribution for a composer pair."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set
    ax = axes[0]
    proba_class0 = result['y_train_proba'][result['y_train'] == 0]
    proba_class1 = result['y_train_proba'][result['y_train'] == 1]
    
    ax.hist(proba_class0, bins=20, alpha=0.6, label=f'{result["composer1"]}', color='blue', edgecolor='black')
    ax.hist(proba_class1, bins=20, alpha=0.6, label=f'{result["composer2"]}', color='red', edgecolor='black')
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Training Set', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Test set
    ax = axes[1]
    proba_class0 = result['y_test_proba'][result['y_test'] == 0]
    proba_class1 = result['y_test_proba'][result['y_test'] == 1]
    
    ax.hist(proba_class0, bins=20, alpha=0.6, label=f'{result["composer1"]}', color='blue', edgecolor='black')
    ax.hist(proba_class1, bins=20, alpha=0.6, label=f'{result["composer2"]}', color='red', edgecolor='black')
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Test Set', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle(f'{result["composer1"]} vs {result["composer2"]}\n'
                 f'Test Balanced Accuracy: {result["balanced_accuracy"]:.4f} ({result["balanced_accuracy"]*100:.2f}%)',
                 fontsize=14)
    
    plt.tight_layout()
    filename = f'figures/logistic_{result["composer1"]}_vs_{result["composer2"]}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def create_composer_matrix_logistic(X, Y, top_n=3):
    """Create a square matrix of logistic regression accuracies."""
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
            result = binary_classify_pair_logistic(X, Y, composer1, composer2)
            
            if result:
                accuracy_matrix[i, j] = result['balanced_accuracy']
                accuracy_matrix[j, i] = result['balanced_accuracy']  # Symmetric
                pair_results.append(result)
                print(f"  Balanced Accuracy: {result['balanced_accuracy']:.4f} ({result['balanced_accuracy']*100:.2f}%)")
                print(f"  Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1']:.4f}")
                
                # Plot probability distribution
                plot_probability_distribution(result, len(pair_results))
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
    ax.set_title('Logistic Regression: Binary Classification Balanced Accuracy\n(Normalized for Class Imbalance)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('figures/logistic_composer_matrix.png', dpi=150, bbox_inches='tight')
    print("\nMatrix visualization saved as 'figures/logistic_composer_matrix.png'")
    plt.close()


def main():
    print("=" * 80)
    print("Logistic Regression: Composer Binary Classification Matrix")
    print("=" * 80)
    
    # Load data
    print("\nLoading preprocessed data...")
    X, Y, feature_names = load_preprocessed_data("data")
    print(f"Loaded {len(X)} segments with {len(feature_names)} features")
    
    # Create matrix
    accuracy_matrix, composer_names, pair_results = create_composer_matrix_logistic(X, Y, top_n=3)
    
    # Print matrix
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION: BALANCED ACCURACY MATRIX")
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
    print("  - logistic_composer_matrix.png (accuracy matrix)")
    for result in pair_results:
        print(f"  - logistic_{result['composer1']}_vs_{result['composer2']}.png")
    print("=" * 80)
    
    return accuracy_matrix, composer_names, pair_results


if __name__ == "__main__":
    np.random.seed(42)
    accuracy_matrix, composer_names, pair_results = main()

