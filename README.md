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

## Project Structure

- **preprocess_midi.py** - Main preprocessing pipeline
- **composer_matrix.py** - Binary classification matrix generator
- **load_data.py** - Utilities to load preprocessed data
- **data/** - Preprocessed feature matrices and labels

## Dataset Information

- Dataset: [drengskapur/midi-classical-music](https://huggingface.co/datasets/drengskapur/midi-classical-music)
- Size: roughly 4,800 MIDI files
- Preprocessed: 10 features per 30-second segment
