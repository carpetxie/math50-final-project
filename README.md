# MIDI Classical Music Dataset Import

This project imports the [midi-classical-music](https://huggingface.co/datasets/drengskapur/midi-classical-music) dataset from Hugging Face.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the import script:
```bash
python import_dataset.py
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("drengskapur/midi-classical-music")

# Access the train split
train_data = dataset['train']

# Get file names
file_names = train_data['file_name']

# Access a specific example
example = train_data[0]
```

## Dataset Information

- **Dataset**: drengskapur/midi-classical-music
- **Size**: ~4.8k rows in the train split
- **Features**: Contains MIDI file names for classical music pieces

