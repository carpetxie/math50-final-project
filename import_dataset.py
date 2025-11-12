"""
Script to import the midi-classical-music dataset from Hugging Face.
Dataset: drengskapur/midi-classical-music
"""

from datasets import load_dataset
import mido
from huggingface_hub import hf_hub_download

def import_midi_dataset():
    """
    Load the midi-classical-music dataset from Hugging Face.
    
    Returns:
        Dataset: The loaded dataset
    """
    print("Loading midi-classical-music dataset from Hugging Face...")
    dataset = load_dataset("drengskapur/midi-classical-music")
    
    print(f"\nDataset loaded successfully!")
    print(f"Dataset splits: {list(dataset.keys())}")
    
    # Print information about the train split
    if 'train' in dataset:
        train_split = dataset['train']
        print(f"\nTrain split info:")
        print(f"  Number of examples: {len(train_split)}")
        print(f"  Features: {train_split.features}")
        
        # Show the metadata structure (schema) of each example
        print(f"\n{'='*80}")
        print("DATASET METADATA STRUCTURE")
        print(f"{'='*80}")
        print(f"\nEach example in the dataset contains the following fields:")
        print("-" * 80)
        
        # Get the first example to show structure
        if len(train_split) > 0:
            first_example = train_split[0]
            print(f"\nField names and types:")
            for field_name, field_type in train_split.features.items():
                print(f"  - {field_name}: {type(field_type).__name__}")
                if hasattr(field_type, 'dtype'):
                    print(f"    Data type: {field_type.dtype}")
                if hasattr(field_type, '_type'):
                    print(f"    Type: {field_type._type}")
            
            print(f"\n\nExample of actual data structure (first example):")
            print("-" * 80)
            for key, value in first_example.items():
                print(f"  {key}:")
                print(f"    Type: {type(value).__name__}")
                print(f"    Value: {value}")
                if isinstance(value, (list, dict)):
                    print(f"    Length/Keys: {len(value)}")
            print("-" * 80)
        
        # Show a few examples with detailed MIDI data
        print(f"\nFirst 5 examples (with MIDI file contents):")
        print("=" * 80)
        for i in range(min(5, len(train_split))):
            example = train_split[i]
            print(f"\nExample {i+1}:")
            print("-" * 80)
            # Print file name
            file_name = example.get('file_name', 'N/A')
            print(f"  File name: {file_name}")
            
            # Try to download and parse the MIDI file
            try:
                # Download the MIDI file from Hugging Face
                midi_path = hf_hub_download(
                    repo_id="drengskapur/midi-classical-music",
                    filename=file_name,
                    repo_type="dataset"
                )
                
                # Parse the MIDI file
                mid = mido.MidiFile(midi_path)
                
                print(f"\n  MIDI File Information:")
                print(f"    - Type: {mid.type}")
                print(f"    - Ticks per beat: {mid.ticks_per_beat}")
                print(f"    - Length (seconds): {mid.length:.2f}")
                print(f"    - Number of tracks: {len(mid.tracks)}")
                
                # Show track information
                print(f"\n  Track Details:")
                for track_idx, track in enumerate(mid.tracks):
                    print(f"    Track {track_idx}: '{track.name}' ({len(track)} messages)")
                    
                    # Count different message types in first track
                    if track_idx == 0:
                        note_on_count = sum(1 for msg in track if msg.type == 'note_on')
                        note_off_count = sum(1 for msg in track if msg.type == 'note_off')
                        program_change = sum(1 for msg in track if msg.type == 'program_change')
                        print(f"      - Note On messages: {note_on_count}")
                        print(f"      - Note Off messages: {note_off_count}")
                        print(f"      - Program Change messages: {program_change}")
                        
                        # Show first few note messages
                        print(f"\n    First 10 messages in Track {track_idx}:")
                        msg_count = 0
                        for msg in track:
                            if msg_count >= 10:
                                break
                            if msg.type in ['note_on', 'note_off', 'program_change', 'set_tempo']:
                                print(f"      {msg}")
                                msg_count += 1
                
                # Extract and show some note data
                print(f"\n  Sample Note Data (first 20 notes):")
                note_count = 0
                # Helper function to convert MIDI note number to note name
                def midi_to_note_name(note_num):
                    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    octave = (note_num // 12) - 1
                    note = notes[note_num % 12]
                    return f"{note}{octave}"
                
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == 'note_on' and msg.velocity > 0:
                            note_name = midi_to_note_name(msg.note)
                            print(f"    Note: {note_name} (MIDI {msg.note}), Velocity: {msg.velocity}, Time: {msg.time}")
                            note_count += 1
                            if note_count >= 20:
                                break
                    if note_count >= 20:
                        break
                        
            except Exception as e:
                print(f"  Error loading MIDI file: {e}")
                print(f"  (File may not be available or accessible)")
            
            print("-" * 80)
    
    return dataset

if __name__ == "__main__":
    dataset = import_midi_dataset()
    
    # Access the dataset
    # Example: dataset['train'][0] to get the first example
    # Example: dataset['train']['file_name'] to get all file names

