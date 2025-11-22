"""
Preprocess MIDI files into feature vectors for composer classification.
"""

import numpy as np
import mido
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pathlib import Path
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class MIDIEvent:
    def __init__(self, note, velocity, start_time, end_time, track_idx):
        self.note = note
        self.velocity = velocity
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.track_idx = track_idx


def extract_midi_events(midi_path: str) -> Tuple[List[MIDIEvent], float]:
    mid = mido.MidiFile(midi_path)
    events = []
    
    # Track active notes: key = (track_idx, note_number), value = (start_time, velocity)
    active_notes = {}
    
    for track_idx, track in enumerate(mid.tracks):
        absolute_time_ticks = 0  # Cumulative time in ticks
        
        for msg in track:
            # Update absolute time
            absolute_time_ticks += msg.time
            
            # Convert ticks to seconds
            absolute_time_seconds = mido.tick2second(
                absolute_time_ticks, 
                mid.ticks_per_beat, 
                mido.bpm2tempo(120)  # Default tempo, will be overridden by tempo messages
            )
            
            # Handle note_on events (velocity > 0 means note starts)
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.note)
                active_notes[key] = (absolute_time_seconds, msg.velocity)
            
            # Handle note_off events or note_on with velocity 0
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.note)
                if key in active_notes:
                    start_time, velocity = active_notes.pop(key)
                    event = MIDIEvent(
                        note=msg.note,
                        velocity=velocity,
                        start_time=start_time,
                        end_time=absolute_time_seconds,
                        track_idx=track_idx
                    )
                    events.append(event)
    
    # Get total length
    total_length = mid.length
    
    return events, total_length


def segment_events(events: List[MIDIEvent], total_length: float, segment_duration: float = 30.0) -> List[List[MIDIEvent]]:
    num_segments = max(1, int(np.ceil(total_length / segment_duration)))
    segments = [[] for _ in range(num_segments)]
    
    for event in events:
        # Assign event to segment based on its start time
        segment_idx = int(event.start_time / segment_duration)
        if segment_idx < num_segments:
            segments[segment_idx].append(event)
    
    return segments


def extract_features(segment: List[MIDIEvent]) -> np.ndarray:
    if len(segment) == 0:
        return np.zeros(10)
    
    notes = np.array([e.note for e in segment])
    velocities = np.array([e.velocity for e in segment])
    start_times = np.array([e.start_time for e in segment])
    durations = np.array([e.duration for e in segment])
    
    sorted_indices = np.argsort(start_times)
    sorted_start_times = start_times[sorted_indices]
    iois = np.diff(sorted_start_times)
    
    mean_pitch = np.mean(notes)
    pitch_stddev = np.std(notes) if len(notes) > 1 else 0.0
    pitch_range = np.max(notes) - np.min(notes)
    
    note_density = len(segment)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    ioi_stddev = np.std(iois) if len(iois) > 1 else 0.0
    
    if len(iois) > 1:
        num_bins = min(20, max(2, len(iois) // 2))
        hist, _ = np.histogram(iois, bins=num_bins)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        ioi_entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
    else:
        ioi_entropy = 0.0
    
    mean_duration = np.mean(durations)
    mean_velocity = np.mean(velocities)
    velocity_stddev = np.std(velocities) if len(velocities) > 1 else 0.0
    
    features = np.array([
        mean_pitch,
        pitch_stddev,
        pitch_range,
        note_density,
        mean_ioi,
        ioi_stddev,
        ioi_entropy,
        mean_duration,
        mean_velocity,
        velocity_stddev
    ])
    
    return features


def extract_composer_and_piece(file_name: str) -> Tuple[str, str]:
    base_name = Path(file_name).stem
    parts = base_name.split('-', 1)
    if len(parts) == 2:
        composer = parts[0].strip()
        piece = parts[1].strip()
    else:
        composer = "unknown"
        piece = base_name
    return composer, piece


def preprocess_dataset(max_files: int = None, segment_duration: float = 30.0):
    print("Loading MIDI dataset from Hugging Face...")
    dataset = load_dataset("drengskapur/midi-classical-music")
    train_split = dataset['train']
    
    X_list = []
    Y_list = []
    
    num_files = len(train_split) if max_files is None else min(max_files, len(train_split))
    
    print(f"\nProcessing {num_files} MIDI files...")
    print(f"Segment duration: {segment_duration} seconds")
    print("=" * 80)
    
    for i in range(num_files):
        example = train_split[i]
        file_name = example['file_name']
        
        try:
            # Download MIDI file
            midi_path = hf_hub_download(
                repo_id="drengskapur/midi-classical-music",
                filename=file_name,
                repo_type="dataset"
            )
            
            # Extract composer and piece
            composer, piece = extract_composer_and_piece(file_name)
            
            # Extract all note events
            events, total_length = extract_midi_events(midi_path)
            
            if len(events) == 0:
                print(f"  [{i+1}/{num_files}] Skipped {file_name} (no note events)")
                continue
            
            # Segment the events
            segments = segment_events(events, total_length, segment_duration)
            
            # Extract features from each segment
            for seg_idx, segment in enumerate(segments):
                if len(segment) > 0:  # Only include non-empty segments
                    features = extract_features(segment)
                    X_list.append(features)
                    Y_list.append((composer, piece))
            
            num_segments = sum(1 for seg in segments if len(seg) > 0)
            print(f"  [{i+1}/{num_files}] {file_name}")
            print(f"    Composer: {composer}, Piece: {piece}")
            print(f"    Length: {total_length:.1f}s, Events: {len(events)}, Segments: {num_segments}")
            
        except Exception as e:
            print(f"  [{i+1}/{num_files}] Error processing {file_name}: {e}")
            continue
    
    # Convert to numpy array
    X = np.array(X_list)
    Y = Y_list
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print(f"Total segments: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Number of unique composers: {len(set(y[0] for y in Y))}")
    print(f"Number of unique pieces: {len(set(y for y in Y))}")
    
    return X, Y


def save_preprocessed_data(X: np.ndarray, Y: List[Tuple[str, str]], output_dir: str = "data"):
    Path(output_dir).mkdir(exist_ok=True)
    
    np.save(f"{output_dir}/X_features.npy", X)
    with open(f"{output_dir}/Y_labels.pkl", "wb") as f:
        pickle.dump(Y, f)
    feature_names = [
        'mean_pitch',
        'pitch_stddev',
        'pitch_range',
        'note_density',
        'mean_ioi',
        'ioi_stddev',
        'ioi_entropy',
        'mean_duration',
        'mean_velocity',
        'velocity_stddev'
    ]
    with open(f"{output_dir}/feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    
    print(f"\nData saved to {output_dir}/")
    print(f"  - X_features.npy: {X.shape}")
    print(f"  - Y_labels.pkl: {len(Y)} labels")
    print(f"  - feature_names.txt: 10 features")


if __name__ == "__main__":
    # Process first 100 files as a test (remove max_files=100 to process all ~4800 files)
    X, Y = preprocess_dataset(max_files=100, segment_duration=30.0)
    
    # Save the preprocessed data
    save_preprocessed_data(X, Y)
    
    # Print some statistics
    print("\n" + "=" * 80)
    print("Feature Statistics:")
    print("-" * 80)
    feature_names = [
        'mean_pitch', 'pitch_stddev', 'pitch_range',
        'note_density', 'mean_ioi', 'ioi_stddev', 'ioi_entropy',
        'mean_duration', 'mean_velocity', 'velocity_stddev'
    ]
    
    for i, name in enumerate(feature_names):
        print(f"{name:20s}: mean={X[:, i].mean():8.2f}, std={X[:, i].std():8.2f}, "
              f"min={X[:, i].min():8.2f}, max={X[:, i].max():8.2f}")
    
    print("\n" + "=" * 80)
    print("Sample labels (first 10):")
    for i in range(min(10, len(Y))):
        print(f"  {i+1}. Composer: {Y[i][0]:20s} Piece: {Y[i][1]}")

