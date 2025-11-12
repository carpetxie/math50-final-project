# MIDI Classical Music Composer Classification

This project preprocesses MIDI classical music data for composer classification using linear models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Preprocess MIDI Files

```bash
python preprocess_midi.py
```

This will:
- Load MIDI files from Hugging Face
- Segment pieces into 30-second chunks
- Extract 9 musical features per segment
- Save to `data/X_features.npy` and `data/Y_labels.pkl`

### 2. View Preprocessed Data

```bash
python load_data.py
```

### 3. Explore Dataset (Optional)

```bash
python import_dataset.py
```

## Project Structure

- **`preprocess_midi.py`** - Main preprocessing pipeline
- **`load_data.py`** - Utilities to load and inspect preprocessed data
- **`import_dataset.py`** - Dataset exploration script
- **`data/`** - Preprocessed feature matrices and labels
- **`PREPROCESSING_README.md`** - Detailed preprocessing documentation
- **`MIDI_DATA_EXPLANATION.md`** - MIDI format reference

## Dataset Information

- **Dataset**: [drengskapur/midi-classical-music](https://huggingface.co/datasets/drengskapur/midi-classical-music)
- **Size**: ~4,800 MIDI files
- **Preprocessed**: 9 features per 30-second segment

