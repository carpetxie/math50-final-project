# MIDI Classical Music Composer Classification

This project preprocesses MIDI classical music data and performs binary composer classification using least squares linear algebra.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Preprocess MIDI Files

```bash
python preprocess_midi.py
```

This downloads MIDI files from Hugging Face, segments pieces into 30-second chunks, extracts 10 musical features per segment, and saves to `data/X_features.npy` and `data/Y_labels.pkl`.

By default, it processes the first 100 files. To process all ~4,800 files, edit `preprocess_midi.py` and remove the `max_files=100` parameter.

### Step 2: Generate Composer Classification Matrix

```bash
python composer_matrix.py
```

This creates a 3×3 accuracy matrix for the top 3 composers (albeniz, bach, alkan) and generates:
- `composer_matrix.png` - Accuracy matrix heatmap
- `threshold_albeniz_vs_bach.png` - Threshold optimization graph
- `threshold_albeniz_vs_alkan.png` - Threshold optimization graph
- `threshold_bach_vs_alkan.png` - Threshold optimization graph

Each pair uses least squares to solve **Ax = b** where A is the feature matrix, x is the weight vector, and b is the binary labels.

### Step 3: Comprehensive Regression Analysis (Units 4-6)

```bash
python regression_analysis.py
```

This performs comprehensive linear regression analysis incorporating mathematical concepts from Units 4-6:

**Unit 4: Multiple Predictor Regression**
- Multiple predictor linear regression: Y = β₀ + Σᵢ βᵢXᵢ + ε
- Coefficient interpretation: βᵢ represents change in Y when Xᵢ increases by 1, holding all other predictors constant
- Correlation matrix analysis to understand multicollinearity
- R² interpretation: proportion of variance explained by all predictors
- Comparison of single-predictor vs multiple-predictor coefficients

**Unit 5: Nonlinear Models & Bias-Variance Tradeoff**
- Interaction terms: Y = β₀ + Σᵢ βᵢXᵢ + Σᵢⱼ JᵢⱼXᵢXⱼ + ε
- Residual plots (residuals vs predicted values, not vs actual Y)
- Bias-variance tradeoff analysis showing training vs test error as complexity increases

**Unit 6: Regularization & Bayesian Perspective**
- Ridge regression (L2 regularization): minimizes ||y - Xβ||² + λ||β||²
- Connection to Bayesian regression with Normal(0, τ²) priors on β
- Coefficient shrinkage analysis

Generates:
- `correlation_matrix.png` - Feature correlation heatmap
- `residual_plots.png` - Residual analysis (correct vs incorrect plots)
- `bias_variance_tradeoff.png` - Training vs test error by model complexity
- `ridge_regression.png` - R² and coefficient norms vs regularization parameter

### Step 4: View Comprehensive Results Summary

```bash
python comprehensive_results_summary.py
```

This aggregates and summarizes ALL analyses:
- Threshold optimization results
- Feature ablation study results
- Correlation matrix findings
- Multi-output regression performance
- Bias-variance tradeoff analysis
- Ridge regression results
- Residual analysis

Provides a complete overview of composer separation and model performance.

## Project Structure

- **preprocess_midi.py** - Main preprocessing pipeline
- **composer_matrix.py** - Binary classification matrix generator
- **regression_analysis.py** - Comprehensive regression analysis (Units 4-6)
- **feature_ablation.py** - Feature importance via ablation study
- **comprehensive_results_summary.py** - Aggregates all analysis results
- **load_data.py** - Utilities to load preprocessed data
- **data/** - Preprocessed feature matrices and labels

## Dataset Information

- Dataset: [drengskapur/midi-classical-music](https://huggingface.co/datasets/drengskapur/midi-classical-music)
- Size: roughly 4,800 MIDI files
- Preprocessed: 10 features per 30-second segment
