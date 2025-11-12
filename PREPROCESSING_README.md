# MIDI Data Preprocessing for Composer Classification

## Overview

This preprocessing pipeline converts MIDI files into feature vectors suitable for linear classification models. The approach follows best practices for music information retrieval (MIR) by extracting meaningful statistical features from fixed-duration segments.

## Approach

### 1. Segmentation Strategy

Each MIDI piece is divided into **30-second segments** to solve the small-dataset problem:

- **Why 30 seconds?** This duration captures enough musical context (typically 1-2 phrases) while maximizing the number of training samples.
- **Example:** An 8-minute piece (480 seconds) → 16 segments of 30 seconds each
- **Benefit:** From ~4,800 pieces, we generate thousands of segments for training

### 2. Feature Extraction

For each 30-second segment, we extract **9 numerical features** organized into three categories:

#### 🎹 Pitch Features (3 features)

1. **`mean_pitch`**: Average MIDI note number (0-127)
   - Captures the register/tessitura of the music
   - Example: High values → treble register, Low values → bass register

2. **`pitch_stddev`**: Standard deviation of MIDI note numbers
   - Measures melodic/harmonic variability
   - High value → wide range, jumpy melodies (e.g., arpeggios)
   - Low value → narrow range, stable melodies

3. **`pitch_range`**: Difference between highest and lowest note
   - Captures the overall pitch span
   - Useful for distinguishing instrument-writing styles

#### 🥁 Rhythmic Features (4 features)

4. **`note_density`**: Total count of notes in the segment
   - **Most powerful feature** for distinguishing styles
   - Fast, busy pieces (Vivaldi) → high density
   - Slow, lyrical pieces → low density

5. **`mean_ioi`**: Average inter-onset interval in seconds
   - IOI = time between consecutive note starts
   - Direct measure of tempo
   - Small IOI → fast tempo, Large IOI → slow tempo

6. **`ioi_stddev`**: Standard deviation of inter-onset intervals
   - Measures rhythmic regularity
   - Low value → steady, motoric rhythm
   - High value → complex/flexible rhythm (rubato)

7. **`mean_duration`**: Average note duration in seconds
   - Captures articulation style
   - Short duration → staccato
   - Long duration → legato

#### 🔊 Dynamic Features (2 features)

8. **`mean_velocity`**: Average note velocity (0-127)
   - Measures overall loudness

9. **`velocity_stddev`**: Standard deviation of velocity
   - **Excellent proxy for dynamic range**
   - High value → expressive (soft to loud)
   - Low value → flat dynamics (e.g., harpsichord)

## Data Format

### X (Feature Matrix)
- **Type:** NumPy array
- **Shape:** `(n_segments, 9)`
- **Example:** `(385, 9)` from 100 files

### Y (Labels)
- **Type:** List of tuples
- **Format:** `(composer_name, piece_name)`
- **Example:** `('bach', 'bwv001-_400_chorales-003706b')`
- **Length:** `n_segments`

### Feature Names
```
1. mean_pitch
2. pitch_stddev
3. pitch_range
4. note_density
5. mean_ioi
6. ioi_stddev
7. mean_duration
8. mean_velocity
9. velocity_stddev
```

## Usage

### Step 1: Preprocess MIDI Files

```python
from preprocess_midi import preprocess_dataset, save_preprocessed_data

# Process all files (or use max_files=N for testing)
X, Y = preprocess_dataset(max_files=100, segment_duration=30.0)

# Save to disk
save_preprocessed_data(X, Y, output_dir="data")
```

### Step 2: Load Preprocessed Data

```python
from load_data import load_preprocessed_data, get_composer_labels

# Load data
X, Y, feature_names = load_preprocessed_data("data")

# Extract just composer labels for classification
composer_labels = get_composer_labels(Y)
```

### Step 3: Train a Linear Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Get composer labels
y = get_composer_labels(Y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (important for linear models!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## File Structure

```
math50-project/
├── preprocess_midi.py          # Main preprocessing script
├── load_data.py                # Data loading utilities
├── import_dataset.py           # Dataset exploration
├── data/                       # Output directory
│   ├── X_features.npy         # Feature matrix
│   ├── Y_labels.pkl           # Labels (composer, piece)
│   └── feature_names.txt      # Feature names
└── PREPROCESSING_README.md    # This file
```

## Example Output

From 100 MIDI files, we generated:

```
Total segments: 385
Feature shape: (385, 9)
Unique composers: 7
Unique pieces: 100

Composer Distribution:
  albeniz  : 130 segments (33.8%)
  bach     : 108 segments (28.1%)
  alkan    :  80 segments (20.8%)
  bacewitz :  28 segments ( 7.3%)
  arensky  :  19 segments ( 4.9%)
  ambroise :  15 segments ( 3.9%)
  arndt    :   5 segments ( 1.3%)
```

## Feature Statistics (from 100 files)

| Feature          | Mean   | Std   | Min   | Max    |
|------------------|--------|-------|-------|--------|
| mean_pitch       | 62.99  | 5.14  | 40.88 | 76.83  |
| pitch_stddev     | 10.63  | 3.00  | 4.45  | 23.05  |
| pitch_range      | 49.56  | 14.29 | 13.00 | 96.00  |
| note_density     | 330.65 | 240.72| 4.00  | 1460.00|
| mean_ioi         | 0.10   | 0.05  | 0.00  | 0.44   |
| ioi_stddev       | 0.15   | 0.10  | 0.00  | 0.79   |
| mean_duration    | 0.36   | 0.25  | 0.04  | 2.91   |
| mean_velocity    | 77.39  | 20.08 | 26.93 | 118.57 |
| velocity_stddev  | 10.05  | 8.73  | 0.00  | 35.17  |

## Why Linear Models?

Linear models (like Logistic Regression) are perfect for this task because:

1. **Interpretable**: You can see which features matter most
2. **Fast**: Trains quickly even on thousands of segments
3. **Robust**: Works well with ~9 features and hundreds of samples
4. **Effective**: These hand-crafted features capture musical style well

## Next Steps

1. **Scale up**: Process all ~4,800 files instead of just 100
2. **Binary classification**: Choose 2 composers and train a model
3. **Multi-class**: Classify among all composers
4. **Feature importance**: Analyze which features distinguish composers best
5. **Cross-validation**: Use k-fold CV for robust evaluation

## Technical Details

### Time Conversion

MIDI stores timing in "ticks" (integer units). We convert to seconds:

```python
seconds = mido.tick2second(ticks, ticks_per_beat, tempo)
```

### Note Duration Calculation

1. `note_on` event (velocity > 0) → note starts
2. `note_off` event (or `note_on` with velocity=0) → note ends
3. Duration = end_time - start_time

### Segment Assignment

Each note is assigned to a segment based on its **start time**:

```python
segment_idx = int(note.start_time / segment_duration)
```

## References

- Dataset: [drengskapur/midi-classical-music](https://huggingface.co/datasets/drengskapur/midi-classical-music) on Hugging Face
- MIDI format: [MIDI.org](https://www.midi.org/)
- Feature extraction: Standard MIR techniques for symbolic music analysis

