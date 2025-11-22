# Classical Composer Prediction via Musical Features

**Repository:** https://github.com/carpetxie/math50-final-project  
**Authors:** Eddie Bae, Jeffrey Xie, Manraaj Singh, Warren Huang

## Hypothesis

The hypothesis of this project is that a linear combination of musical parameters significantly predicts the target variable of the composer. In other words, we think composers have distinct "musical fingerprints" that can be captured through quantifiable features like pitch, rhythm, and dynamics—and that these fingerprints are linearly separable enough for regression-based classification.

## Literature Review

Automatically identifying a composer from a musical piece is a central challenge in Music Information Retrieval (MIR). Our project engages with this problem by adopting a "global feature" approach, as supported by Herremans, Martens, and Sörensen (2016). In their study, they were able to classify pieces by Bach, Haydn, and Beethoven through 12 statistical features from symbolic music data, which validates our method of creating a musical "fingerprint" from MIDI segments.

While other studies have shown success with sequential "local feature" models like n-grams, the global approach is specifically chosen for its interpretability. As the MIR literature emphasizes, "strategies based on hand-crafted mid-level features are still of relevance" precisely because they "allow interpretable and controllable systems" that reveal *why* a classification was made, a goal often obscured in complex "black box" models (Chowdhury et al., 2022).

Building on this foundation, our feature set includes sophisticated metrics, such as `ioi_entropy` (inter-onset interval entropy), to capture rhythmic complexity. This concept is well-supported by studies like Febres & Jaffe (2017), who propose viewing music through its "entropy content" and "symbolic diversity" as a powerful method for "music style recognition," and by research such as Gündüz (2023), which explores how entropy is inherently linked to musical order, complexity, and even perceived instability in melodies.

For our classification model, we adopted a one-vs-one (OVO) strategy, decomposing the multi-composer problem into a series of binary classifiers. This approach is a standard and highly effective technique for multiclass classification, a conclusion supported by foundational comparative studies in the field (Hsu & Lin, 2002). By grounding our work in established methods (global features, entropy, and OVO classification), our project provides a robust analysis that quantifies the stylistic "fingerprints" of different composers.

## Dataset

For our dataset, we used [drengskapur's Hugging Face MIDI files](https://huggingface.co/datasets/drengskapur/midi-classical-music) on various classical music pieces—about 4,800 MIDI files in total. For analysis, 10 parameters were used:

1. **mean_pitch** - Average MIDI note number
2. **pitch_stddev** - Standard deviation of pitches
3. **pitch_range** - Difference between highest and lowest note
4. **note_density** - Total count of notes in segment
5. **mean_ioi** - Mean inter-onset interval (time between note starts)
6. **ioi_stddev** - Standard deviation of inter-onset intervals
7. **ioi_entropy** - Shannon entropy of IOI distribution (rhythmic complexity)
8. **mean_duration** - Average note duration
9. **mean_velocity** - Average note velocity (dynamics)
10. **velocity_stddev** - Standard deviation of velocity

These parameters are standards in evaluating music, but for inter-onset entropy (IOI entropy), ideas of Shannon's entropy were used, which dictates rhythmic complexity in a piece: higher entropy is reflected in more diverse IOI distribution (Gündüz et al. 2023).

Then, using these parameters, they were used to distinguish 3 composers: **Albeniz, Bach, and Alkan**. These composers were chosen because they had the most amount of data/pieces available for analysis.

To increase our dataset size, we split each piece into segments of 30 seconds. This roughly quadrupled our dataset to **385 datapoints**. With the 10 features mentioned above, our input data has a shape of (385, 10). Our labels consisted of tuples with the format of `(composer, piece_name)`.

## Methodology

Our analysis includes four modeling components:

### 1. Least-Squares Linear Regression (Binary Classification)

Our first experiment was a simple OVO classification between each pair of composers from the set of Albeniz, Bach, and Alkan. This was built via a **least squares linear regression** model with binary labels for either composer. Features were standardized to zero mean and unit variance before training. The model finds the optimal parameter weightings that minimize the squared error between the linear combination of features and the binary labels.

However, since least squares produce continuous output values rather than binary predictions, we need to find a threshold such that if the output value exceeds that threshold, the classification is the second composer, and if it falls below, the classification is the first composer. We select the threshold that maximizes balanced accuracy on the training set, and we evaluate performance using balanced accuracy to account for class imbalance.

The results are shown in the figures. Interestingly, OVO classifications including Bach all exceed 0.9 accuracy yet Alkan vs. Albeniz only yields a 0.682 accuracy.

### 2. Feature Ablation Study

The second experiment tests ablations on each of the ten features, still utilizing the least squares linear regression model from above. However, instead of an OVO, we run an **One versus Rest (OVR)** classification for a balanced accuracy assessment on the top three composers (Albeniz, Bach, and Alkan).

For each composer, we first establish a baseline accuracy using all features, then remove each feature individually and compare the resulting accuracy against this baseline. Features were standardized to zero mean and unit variance before training, consistent with the first experiment. Interestingly, some features, when removed, would *increase* the accuracy for some composers—suggesting those features may add noise rather than signal.

### 3. Logistic Regression

For our third experiment, we repeated the OVO classification between each pair of composers (Albeniz, Bach, and Alkan) using **logistic regression** instead of least squares. Features were standardized to zero mean and unit variance before training, consistent with the previous experiments.

Unlike least squares, logistic regression models the probability of class membership directly through the logistic function, which naturally constrains outputs to the range [0, 1]. This eliminates the need for threshold optimization, as the decision boundary is fixed at 0.5 probability. The model finds optimal parameter weightings that maximize the likelihood of the observed binary labels.

We evaluate performance using balanced accuracy to account for class imbalance. The accuracy values are nearly identical to those from the least squares approach, with Albeniz vs Bach at 0.943, Bach vs Alkan at 0.977, and Albeniz vs Alkan at 0.688. This similarity is expected, as both methods are linear classifiers that differ primarily in their optimization objectives and output interpretation.

### 4. Classical Linear Regression (Multiple Predictors & Residuals)

In addition to our classifier-based experiments, we also apply the classical linear regression tools required by the course. Unlike the previous sections, the goal here is not to build a composer classifier but to use our dataset to illustrate **multiple-predictor regression, residual diagnostics, ridge regression, and the bias–variance tradeoff**.

Through the `regression_analysis.py` script, a multiple-predictor linear regression was fit using all 10 features at once, where:

**Y = β₀ + β₁X₁ + … + β₁₀X₁₀ + ε**

with Y as a binary composer label and the Xᵢ as the musical features. Compared to single-predictor regressions, the multiple-predictor model controls for correlations among features. For example, pitch standard deviation and pitch range are correlated, and a simple regression can mix their effects. The multiple-predictor model separates these contributions so each βᵢ reflects the direct effect of its feature.

The residual plot shows a limitation of using linear regression for binary outcomes: because Y is binary, the residuals fall on two diagonal lines. These lines come from algebra, not model behavior, and they make the residual plot uninformative for diagnosing issues like curvature or heteroskedasticity. This limitation is one reason logistic regression is usually preferred for binary outcomes.

### 5. Ridge Regression and Bias-Variance

Beyond the multiple-predictor fit, we also study model complexity using **ridge regression** and the **bias–variance tradeoff** to understand how regularization affects performance.

Ridge regression adds an L2 penalty to the least squares objective:

**minimize ||y - Xβ||² + λ||β||²**

This is equivalent to Bayesian regression with Normal(0, τ²) priors on the coefficients. As λ increases, coefficients shrink toward zero, producing a simpler model. The optimal λ is found by maximizing test R², which balances fitting the training data with generalization.

The bias-variance tradeoff illustrates a fundamental machine learning concept: as model complexity increases, training error decreases (lower bias), while test error eventually increases (higher variance). Since MSE = Variance + Bias², the test MSE forms the expected U-shaped curve. Even though Y is binary-coded, these diagnostics remain valid because they evaluate the continuous fitted predictions ŷ, which are later thresholded for classification.

## Results

While linear regression was used in several forms to connect with class concepts (multiple predictors, residuals, ridge regression, and bias–variance), we also implemented logistic regression as the more appropriate model for binary composer prediction. This gives us a clean comparison: the linear probability model is helpful pedagogically, but logistic regression is better aligned with the underlying statistics of a 0/1 outcome.

The similar performance between the two approaches suggests that the main structure in our feature space is largely **linearly separable**, and that simple linear classifiers capture most of the signal in distinguishing these composers.

**Key Findings:**
- **Binary classification:** 94%+ balanced accuracy for Bach vs Albeniz and Bach vs Alkan
- **Albeniz vs Alkan:** 68% balanced accuracy (moderate, but well above chance)
- **Multi-output regression:** 67.5% classification accuracy (4.7× better than random guessing for 7 classes)
- **Feature importance:** Rhythmic features (IOI, duration) and dynamic features are most distinguishing

This project proves our hypothesis that it is possible to use regression to classify classical music composers from different components of their music.

## Open Questions

However, many open questions remain for discovery:
- What efficacy does different types of musical analyses have on identifying composers?
- How does segment length affect the results (we used 30-second intervals)?
- How does polynomial regression (or other types of nonlinear regression) increase or decrease the accuracy of the prediction model?

## Setup & Usage

### Installation

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Analysis

#### Step 1: Preprocess MIDI Files

```bash
python preprocess_midi.py
```

This downloads MIDI files from Hugging Face, segments pieces into 30-second chunks, extracts 10 musical features per segment, and saves to `data/X_features.npy` and `data/Y_labels.pkl`.

*Note: By default, it processes the first 100 files. To process all ~4,800 files, edit `preprocess_midi.py` and remove the `max_files=100` parameter.*

#### Step 2: Generate Binary Classification Matrix

```bash
python composer_matrix.py
```

This creates a 3×3 accuracy matrix for the top 3 composers and generates threshold optimization graphs showing how classification accuracy changes with different decision thresholds.

**Outputs:**
- `figures/composer_matrix.png` - Accuracy matrix heatmap
- `figures/threshold_*.png` - Threshold optimization graphs for each pair

#### Step 3: Feature Ablation Study

```bash
python feature_ablation.py
```

This removes each feature individually to assess its contribution to classification accuracy.

**Outputs:**
- `figures/feature_ablation_heatmap.png` - Feature importance heatmap

#### Step 4: Logistic Regression Analysis

```bash
python logistic_regression.py
```

This repeats the OVO classification using logistic regression instead of least squares for comparison.

**Outputs:**
- `figures/logistic_composer_matrix.png` - Logistic regression accuracy matrix
- `figures/logistic_*.png` - Decision boundary visualizations for each pair

#### Step 5: Comprehensive Regression Analysis

```bash
python regression_analysis.py
```

This performs comprehensive linear regression analysis incorporating mathematical concepts from the course: multiple-predictor regression, correlation analysis, residual diagnostics, bias-variance tradeoff, and ridge regression.

**Outputs:**
- `figures/correlation_matrix.png` - Feature correlation heatmap
- `figures/residual_plots.png` - Residual analysis (correct vs incorrect)
- `figures/bias_variance_tradeoff.png` - Training vs test error by model complexity
- `figures/ridge_regression.png` - R² and coefficient norms vs regularization parameter

#### Step 6: View Comprehensive Results Summary

```bash
python comprehensive_results_summary.py
```

This aggregates and summarizes ALL analyses, providing a complete overview of composer separation and model performance.

## Project Structure

```
math50-project/
├── data/                           # Preprocessed feature matrices and labels
│   ├── X_features.npy             # Feature matrix (385 segments × 10 features)
│   ├── Y_labels.pkl               # Composer/piece labels
│   └── feature_names.txt          # Feature names
├── figures/                        # All generated plots and visualizations
│   ├── composer_matrix.png
│   ├── threshold_*.png
│   ├── feature_ablation_heatmap.png
│   ├── logistic_*.png
│   ├── correlation_matrix.png
│   ├── residual_plots.png
│   ├── bias_variance_tradeoff.png
│   └── ridge_regression.png
├── preprocess_midi.py             # Main preprocessing pipeline
├── load_data.py                   # Data loading utilities
├── composer_matrix.py             # Binary classification matrix (least squares)
├── feature_ablation.py            # Feature importance via ablation
├── logistic_regression.py         # Logistic regression classifier
├── regression_analysis.py         # Comprehensive regression analysis
├── comprehensive_results_summary.py  # Aggregated results summary
├── REGRESSION_ANALYSIS_EXPLANATION.md  # Detailed explanation of regression analysis
├── RESULTS_ANALYSIS.md            # Verification of results and findings
└── requirements.txt               # Python dependencies
```

## Figures

All generated figures are in the `figures/` directory:

1. **composer_matrix.png** - 3×3 heatmap showing balanced accuracy for each composer pair using least squares
2. **threshold_albeniz_vs_bach.png** - Threshold optimization for Albeniz vs Bach
3. **threshold_albeniz_vs_alkan.png** - Threshold optimization for Albeniz vs Alkan
4. **threshold_bach_vs_alkan.png** - Threshold optimization for Bach vs Alkan
5. **feature_ablation_heatmap.png** - Feature importance via ablation study (OVR)
6. **logistic_composer_matrix.png** - 3×3 heatmap using logistic regression
7. **logistic_albeniz_vs_bach.png** - Logistic regression decision boundary
8. **logistic_albeniz_vs_alkan.png** - Logistic regression decision boundary
9. **logistic_bach_vs_alkan.png** - Logistic regression decision boundary
10. **correlation_matrix.png** - Feature correlation analysis
11. **residual_plots.png** - Residual diagnostics (correct vs incorrect plots)
12. **bias_variance_tradeoff.png** - Training vs test error by model complexity
13. **ridge_regression.png** - Ridge regression performance and coefficient shrinkage

## Closing Thoughts

While linear regression was used in several forms to connect with class concepts (multiple predictors, residuals, ridge regression, and bias–variance), we also implemented logistic regression as the more appropriate model for binary composer prediction. This gives us a clean comparison: the linear probability model is helpful pedagogically, but logistic regression is better aligned with the underlying statistics of a 0/1 outcome.

The similar performance between the two approaches suggests that the main structure in our feature space is largely linearly separable, and that simple linear classifiers capture most of the signal in distinguishing these composers. This proves our hypothesis that it is possible to use regression to classify classical music composers from different components of their music.

## Sources

Chowdhury, A., et al. (2022). How Do You See Me? A Framework for "Musicologist-Friendly" Explanations. *Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR)*.

Drengskapur. (2022). MIDI Classical Music Dataset. *Hugging Face*. Retrieved from https://huggingface.co/datasets/drengskapur/midi-classical-music

Febres, G., & Jaffe, K. (2017). Music viewed by its entropy content: A novel window for comparative analysis. *PLoS ONE 12*(10): e0185757. https://doi.org/10.1371/journal.pone.0185757

Gündüz, Güngör. (2023). "Entropy, energy, and instability in music". *Physica A: Statistical Mechanics and its Applications*, vol. 609, 128365. https://doi.org/10.1016/j.physa.2022.128365

Herremans, D., Martens, D., & Sörensen, K. (2016). "Composer Classification Models for Music-Theory Building." In D. Meredith (Ed.), *Computational Music Analysis* (pp. 369-392). Springer.

Hsu, C. W., & Lin, C. J. (2002). A comparison of methods for multiclass support vector machines. *IEEE Transactions on Neural Networks*, 13(2), 415-425.
