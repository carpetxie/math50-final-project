# math50-final-project

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Launch Jupyter Lab/Notebook and open `notebooks/midi_analysis_template.ipynb`.

## Project layout

- `src/data`: Streaming utilities for Hugging Face MIDI datasets.
- `src/features`: Feature extraction helpers built on `pretty_midi`.
- `notebooks/midi_analysis_template.ipynb`: End-to-end analysis template.
- `data/processed`: Default location for exported feature tables.

## Workflow

The notebook guides you through:

1. Loading MIDI metadata from Hugging Face.
2. Streaming and parsing each `.mid` file on the fly.
3. Extracting rhythmic entropy, pitch-class entropy, polyphony, note density, pitch statistics, and note-duration statistics.
4. Saving feature tables for downstream modeling.
5. Running baseline visualizations and linear regression.
