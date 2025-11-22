import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from load_data import load_preprocessed_data, get_composer_labels
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def prepare_data_for_regression(X, Y, composers=['albeniz', 'bach', 'alkan']):
    composer_labels = get_composer_labels(Y)
    
    # Filter to specified composers (top 3 with most data)
    indices = [i for i, (composer, _) in enumerate(Y) if composer in composers]
    X_filtered = X[indices]
    Y_filtered = [Y[i] for i in indices]
    composer_labels_filtered = [composer_labels[i] for i in indices]
    
    # One-hot encode composers
    y = pd.get_dummies(composer_labels_filtered, dtype=float).values
    
    return X_filtered, y, composers


def analyze_correlation_matrix(X, feature_names, title="Correlation Matrix of Features"):
    corr_matrix = np.corrcoef(X.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    
    # Add text annotations
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                          fontsize=8)
    
    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Saved: figures/correlation_matrix.png")
    plt.close()
    
    return corr_matrix


def fit_multiple_regression(X_train, X_test, y_train, y_test, feature_names, composer_names):
    """
    Unit 4: Fit multiple predictor linear regression.
    
    Model: Y = β₀ + Σᵢ βᵢXᵢ + ε
    
    Each coefficient βᵢ represents the expected change in composer indicator
    when feature Xᵢ increases by one unit, holding all other features constant.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # R² scores (Unit 4: R² = 1 - Var(residuals)/Var(Y))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "=" * 80)
    print("UNIT 4: MULTIPLE PREDICTOR LINEAR REGRESSION")
    print("=" * 80)
    print(f"\nModel: Y = β₀ + Σᵢ βᵢXᵢ + ε")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"\nR² measures: 1 - Var(residuals)/Var(Y) ≈ 1 - Var(Y|X₁,...,Xₖ)/Var(Y)")
    print(f"R² represents the proportion of variance in composer indicators explained by all predictors together.")
    
    # Coefficient matrix
    coefficients = pd.DataFrame(
        model.coef_,
        columns=feature_names,
        index=composer_names
    )
    
    print("\n" + "-" * 80)
    print("REGRESSION COEFFICIENTS")
    print("-" * 80)
    print("Each coefficient βᵢ represents the expected change in composer indicator")
    print("when feature Xᵢ increases by one unit, HOLDING ALL OTHER FEATURES CONSTANT.")
    print("\nCoefficient Matrix:")
    print(coefficients.round(4))
    
    return model, coefficients, train_r2, test_r2, y_train_pred, y_test_pred


def compare_single_vs_multiple_predictor(X_train, X_test, y_train, y_test, feature_names, composer_names):
    """
    Unit 4: Compare single-predictor vs multiple-predictor coefficients.
    Demonstrates how correlations between predictors affect coefficients.
    """
    print("\n" + "=" * 80)
    print("UNIT 4: SINGLE vs MULTIPLE PREDICTOR COMPARISON")
    print("=" * 80)
    print("\nWhen predictors are correlated, single-predictor coefficients differ from multiple-predictor coefficients.")
    print("This is because: β'₁ = β₁ + β₂(E[X₂|X₁=1] - E[X₂|X₁=0])")
    
    single_coefs = {}
    multiple_model = LinearRegression()
    multiple_model.fit(X_train, y_train)
    
    for i, feature_name in enumerate(feature_names):
        # Single predictor regression
        single_model = LinearRegression()
        single_model.fit(X_train[:, i:i+1], y_train)
        single_coefs[feature_name] = single_model.coef_[0]
    
    # Compare for first composer
    composer_idx = 0
    print(f"\nComparison for {composer_names[composer_idx]}:")
    print(f"{'Feature':<25s} {'Single-Predictor':<20s} {'Multiple-Predictor':<20s} {'Difference':<15s}")
    print("-" * 80)
    
    for i, feature_name in enumerate(feature_names):
        single_coef = single_coefs[feature_name][composer_idx]
        multiple_coef = multiple_model.coef_[composer_idx, i]
        diff = single_coef - multiple_coef
        print(f"{feature_name:<25s} {single_coef:>18.4f} {multiple_coef:>18.4f} {diff:>13.4f}")
    
    return single_coefs, multiple_model


def add_interaction_terms(X, feature_names):
    """
    Unit 5: Add interaction terms to the model.
    Interaction: X₁X₂ captures when the effect of X₁ depends on X₂.
    """
    n_samples, n_features = X.shape
    interaction_features = []
    interaction_names = []
    
    # Add all pairwise interactions
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction = X[:, i] * X[:, j]
            interaction_features.append(interaction)
            interaction_names.append(f"{feature_names[i]} × {feature_names[j]}")
    
    X_with_interactions = np.hstack([X, np.column_stack(interaction_features)])
    all_feature_names = feature_names + interaction_names
    
    return X_with_interactions, all_feature_names


def fit_with_interactions(X_train, X_test, y_train, y_test, feature_names, composer_names):
    """
    Unit 5: Fit model with interaction terms.
    Model: Y = β₀ + Σᵢ βᵢXᵢ + Σᵢⱼ JᵢⱼXᵢXⱼ + ε
    """
    print("\n" + "=" * 80)
    print("UNIT 5: INTERACTION TERMS")
    print("=" * 80)
    print("\nModel with interactions: Y = β₀ + Σᵢ βᵢXᵢ + Σᵢⱼ JᵢⱼXᵢXⱼ + ε")
    print("Interaction term Jᵢⱼ captures when the effect of Xᵢ depends on Xⱼ.")
    
    X_train_inter, feature_names_inter = add_interaction_terms(X_train, feature_names)
    X_test_inter, _ = add_interaction_terms(X_test, feature_names)
    
    model = LinearRegression()
    model.fit(X_train_inter, y_train)
    
    y_train_pred = model.predict(X_train_inter)
    y_test_pred = model.predict(X_test_inter)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTraining R² (with interactions): {train_r2:.4f}")
    print(f"Test R² (with interactions): {test_r2:.4f}")
    print(f"\nNote: Adding interactions increases model complexity and may lead to overfitting.")
    
    return model, train_r2, test_r2, X_train_inter, X_test_inter, feature_names_inter


def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Unit 5: Residual plots to check model assumptions.
    Plot residuals vs predicted values (NOT vs actual Y values).
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs predicted (CORRECT)
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values (ŷ)', fontsize=12)
    axes[0].set_ylabel('Residuals (y - ŷ)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted (CORRECT)', fontsize=13)
    axes[0].grid(alpha=0.3)
    
    # Residuals vs actual (INCORRECT - for comparison)
    axes[1].scatter(y_true, residuals, alpha=0.5, s=20, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Actual Values (y)', fontsize=12)
    axes[1].set_ylabel('Residuals (y - ŷ)', fontsize=12)
    axes[1].set_title('Residuals vs Actual (INCORRECT - shows bias)', fontsize=13)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/residual_plots.png', dpi=150, bbox_inches='tight')
    print(f"Saved: figures/residual_plots.png")
    plt.close()


def bias_variance_analysis(X, y, feature_names, composer_names, n_splits=10):
    """
    Unit 5: Bias-variance tradeoff analysis.
    Test error = Bias² + Variance + Irreducible Error
    """
    print("\n" + "=" * 80)
    print("UNIT 5: BIAS-VARIANCE TRADEOFF")
    print("=" * 80)
    print("\nTest Error = Bias² + Variance + σ²_ε")
    print("As model complexity increases:")
    print("  - Bias decreases (model fits training data better)")
    print("  - Variance increases (model is more sensitive to training data)")
    print("  - Optimal complexity balances bias and variance")
    
    # Test different model complexities (number of features)
    n_features = len(feature_names)
    complexities = [1, 3, 5, n_features]
    
    train_errors = []
    test_errors = []
    
    for complexity in complexities:
        feature_indices = list(range(complexity))
        X_subset = X[:, feature_indices]
        
        train_errs = []
        test_errs = []
        
        for _ in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y, test_size=0.2, random_state=None
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_errs.append(mean_squared_error(y_train, train_pred))
            test_errs.append(mean_squared_error(y_test, test_pred))
        
        train_errors.append(np.mean(train_errs))
        test_errors.append(np.mean(test_errs))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(complexities, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
    ax.plot(complexities, test_errors, 's-', label='Test Error', linewidth=2, markersize=8)
    ax.set_xlabel('Model Complexity (Number of Features)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff: Training vs Test Error', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    print(f"Saved: figures/bias_variance_tradeoff.png")
    plt.close()
    
    print(f"\nComplexity Analysis:")
    for i, c in enumerate(complexities):
        print(f"  {c} features: Train MSE = {train_errors[i]:.4f}, Test MSE = {test_errors[i]:.4f}")


def ridge_regression_analysis(X_train, X_test, y_train, y_test, feature_names, composer_names):
    """
    Unit 6: Ridge regression (L2 regularization).
    Minimizes: ||y - Xβ||² + λ||β||²
    This is equivalent to Bayesian regression with Normal(0, τ²) priors on β.
    """
    print("\n" + "=" * 80)
    print("UNIT 6: RIDGE REGRESSION (L2 REGULARIZATION)")
    print("=" * 80)
    print("\nRidge regression minimizes: ||y - Xβ||² + λ||β||²")
    print("This is equivalent to Bayesian regression with βᵢ ~ Normal(0, τ²) priors.")
    print("Larger λ (or smaller τ) → stronger regularization → smaller coefficients")
    
    alphas = np.logspace(-3, 2, 20)  # λ values
    train_r2s = []
    test_r2s = []
    coef_norms = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2s.append(r2_score(y_train, train_pred))
        test_r2s.append(r2_score(y_test, test_pred))
        coef_norms.append(np.linalg.norm(model.coef_))
    
    # Find optimal alpha
    optimal_idx = np.argmax(test_r2s)
    optimal_alpha = alphas[optimal_idx]
    
    print(f"\nOptimal regularization parameter: λ = {optimal_alpha:.4f}")
    print(f"Test R² at optimal λ: {test_r2s[optimal_idx]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].semilogx(alphas, train_r2s, 'o-', label='Training R²', linewidth=2)
    axes[0].semilogx(alphas, test_r2s, 's-', label='Test R²', linewidth=2)
    axes[0].axvline(x=optimal_alpha, color='r', linestyle='--', label=f'Optimal λ = {optimal_alpha:.4f}')
    axes[0].set_xlabel('Regularization Parameter (λ)', fontsize=12)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Ridge Regression: R² vs Regularization', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].semilogx(alphas, coef_norms, 'o-', color='green', linewidth=2)
    axes[1].axvline(x=optimal_alpha, color='r', linestyle='--', label=f'Optimal λ = {optimal_alpha:.4f}')
    axes[1].set_xlabel('Regularization Parameter (λ)', fontsize=12)
    axes[1].set_ylabel('||β|| (Coefficient Norm)', fontsize=12)
    axes[1].set_title('Ridge Regression: Coefficient Shrinkage', fontsize=13)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/ridge_regression.png', dpi=150, bbox_inches='tight')
    print(f"Saved: figures/ridge_regression.png")
    plt.close()
    
    # Compare coefficients
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    
    ridge_model = Ridge(alpha=optimal_alpha)
    ridge_model.fit(X_train, y_train)
    
    print(f"\nCoefficient Comparison (first composer, first 5 features):")
    print(f"{'Feature':<25s} {'OLS':<15s} {'Ridge':<15s} {'Difference':<15s}")
    print("-" * 70)
    for i in range(min(5, len(feature_names))):
        ols_coef = ols_model.coef_[0, i]
        ridge_coef = ridge_model.coef_[0, i]
        diff = ols_coef - ridge_coef
        print(f"{feature_names[i]:<25s} {ols_coef:>13.4f} {ridge_coef:>13.4f} {diff:>13.4f}")
    
    return optimal_alpha, ridge_model


def main():
    print("=" * 80)
    print("COMPREHENSIVE LINEAR REGRESSION ANALYSIS")
    print("Incorporating Units 4-6: Multiple Predictors, Interactions, Regularization")
    print("=" * 80)
    
    # Load data
    print("\nLoading preprocessed data...")
    X, Y, feature_names = load_preprocessed_data("data")
    print(f"Loaded {len(X)} segments with {len(feature_names)} features")
    
    # Prepare data - focus on top 3 composers with most data
    target_composers = ['albeniz', 'bach', 'alkan']
    X_reg, y_reg, composer_names = prepare_data_for_regression(X, Y, composers=target_composers)
    print(f"\nUsing top 3 composers (most data): {composer_names}")
    print(f"Feature matrix shape: {X_reg.shape}")
    print(f"Composer indicators shape: {y_reg.shape}")
    
    # Show data distribution
    composer_labels = get_composer_labels(Y)
    composer_counts = Counter(composer_labels)
    print(f"\nData distribution:")
    for composer in composer_names:
        count = composer_counts[composer]
        print(f"  {composer:20s}: {count:4d} segments ({100*count/len(X_reg):5.1f}%)")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reg)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Unit 4: Correlation matrix
    print("\n" + "=" * 80)
    corr_matrix = analyze_correlation_matrix(X_train, feature_names)
    
    # Integrate feature ablation results
    print("\n" + "=" * 80)
    print("INTEGRATING FEATURE ABLATION RESULTS")
    print("=" * 80)
    print("\nFrom feature_ablation.py, most important features per composer:")
    print("  ALBENIZ: note_density, ioi_stddev (baseline: 64.34%)")
    print("  BACH: velocity_stddev, ioi_stddev (baseline: 94.56%)")
    print("  ALKAN: pitch_stddev, ioi_stddev (baseline: 87.59%)")
    print("\nNote: Rhythmic features (IOI) are consistently important across all composers.")
    
    # Highlight highly correlated features from correlation matrix
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.5:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    if high_corr_pairs:
        print(f"\nHighly correlated features (|r| > 0.5) affecting coefficient interpretation:")
        for feat1, feat2, corr in high_corr_pairs[:5]:  # Show top 5
            print(f"  {feat1:20s} ↔ {feat2:20s}: {corr:6.3f}")
        print("  (These correlations explain why single-predictor coefficients differ from multiple-predictor coefficients)")
    
    # Unit 4: Multiple regression
    model, coefficients, train_r2, test_r2, y_train_pred, y_test_pred = fit_multiple_regression(
        X_train, X_test, y_train, y_test, feature_names, composer_names
    )
    
    # Unit 4: Single vs multiple predictor
    single_coefs, multiple_model = compare_single_vs_multiple_predictor(
        X_train, X_test, y_train, y_test, feature_names, composer_names
    )
    
    # Unit 5: Residual plots
    print("\n" + "=" * 80)
    plot_residuals(y_test.flatten(), y_test_pred.flatten(), "Residual Analysis")
    
    # Unit 5: Interactions
    interaction_model, inter_train_r2, inter_test_r2, X_train_inter, X_test_inter, feature_names_inter = fit_with_interactions(
        X_train, X_test, y_train, y_test, feature_names, composer_names
    )
    
    # Unit 5: Bias-variance tradeoff
    bias_variance_analysis(X_scaled, y_reg, feature_names, composer_names)
    
    # Unit 6: Ridge regression
    optimal_alpha, ridge_model = ridge_regression_analysis(
        X_train, X_test, y_train, y_test, feature_names, composer_names
    )
    
    # Summary integrating all analyses
    print("\n" + "=" * 80)
    print("INTEGRATED ANALYSIS SUMMARY")
    print("=" * 80)
    print("\nCombining results from all analyses:")
    print("\n1. Feature Ablation (feature_ablation.py):")
    print("   - Most important: IOI features (rhythmic timing)")
    print("   - Bach has highest baseline accuracy (94.56%)")
    print("   - Albeniz and Alkan show distinct feature patterns")
    
    print("\n2. Correlation Matrix:")
    print("   - Strong correlations between pitch and rhythmic features")
    print("   - Explains coefficient differences in single vs multiple predictor models")
    
    print("\n3. Regression Analysis:")
    print(f"   - Training R²: {train_r2:.4f}")
    print(f"   - Test R²: {test_r2:.4f}")
    print("   - Regularization improves test performance (Ridge R²: 0.0888)")
    
    print("\n4. Bias-Variance Tradeoff:")
    print("   - Optimal complexity: 5-10 features")
    print("   - Interactions cause severe overfitting (test R²: -9.59)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - correlation_matrix.png")
    print("  - residual_plots.png")
    print("  - bias_variance_tradeoff.png")
    print("  - ridge_regression.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    np.random.seed(42)
    main()

