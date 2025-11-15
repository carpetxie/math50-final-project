"""Feature extraction helpers for MIDI files using ``pretty_midi``."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pretty_midi
from scipy.stats import entropy


EPS = 1e-12


def _collect_notes(midi: pretty_midi.PrettyMIDI):
    notes = []
    for instrument in midi.instruments:
        notes.extend(instrument.notes)
    return notes


def _note_durations(notes):
    return np.array([note.end - note.start for note in notes], dtype=float)


def _note_intervals(notes):
    if len(notes) < 2:
        return np.array([], dtype=float)
    start_times = sorted(note.start for note in notes)
    return np.diff(start_times)


def compute_rhythmic_entropy(notes) -> float:
    intervals = _note_intervals(notes)
    if intervals.size == 0:
        return 0.0
    hist, _ = np.histogram(intervals, bins="auto", density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    hist = hist / hist.sum()
    return float(entropy(hist, base=2))


def compute_pitch_class_entropy(notes) -> float:
    if not notes:
        return 0.0
    pitch_classes = np.array([note.pitch % 12 for note in notes], dtype=int)
    counts = np.bincount(pitch_classes, minlength=12).astype(float)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    return float(entropy(nz, base=2))


def compute_polyphony(midi: pretty_midi.PrettyMIDI) -> float:
    times = np.linspace(midi.get_onsets().min() if midi.get_onsets().size else 0.0,
                        midi.get_end_time(), num=200)
    if times.size == 0:
        return 0.0

    actives = []
    for t in times:
        active_notes = 0
        for instrument in midi.instruments:
            for note in instrument.notes:
                if note.start <= t < note.end:
                    active_notes += 1
        actives.append(active_notes)

    if not actives:
        return 0.0
    return float(np.mean(actives))


def compute_note_density(midi: pretty_midi.PrettyMIDI) -> float:
    duration = midi.get_end_time()
    if duration <= EPS:
        return 0.0
    total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
    return float(total_notes / duration)


def extract_midi_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """Extract a dictionary of musical features from a MIDI object."""

    notes = _collect_notes(midi)

    if not notes:
        return {
            "duration_sec": float(midi.get_end_time()),
            "tempo_mean": 0.0,
            "tempo_std": 0.0,
            "pitch_range": 0.0,
            "pitch_mean": 0.0,
            "pitch_variance": 0.0,
            "note_duration_mean": 0.0,
            "note_duration_variance": 0.0,
            "note_density": 0.0,
            "polyphony": 0.0,
            "rhythmic_entropy": 0.0,
            "pitch_class_entropy": 0.0,
        }

    duration_sec = float(midi.get_end_time())
    tempo_times, tempo_values = midi.get_tempo_changes()
    if tempo_values.size == 0:
        tempo_values = np.array([midi.estimate_tempo()])
    tempo_mean = float(np.mean(tempo_values))
    tempo_std = float(np.std(tempo_values))

    pitches = np.array([note.pitch for note in notes], dtype=int)
    pitch_range = float(pitches.max() - pitches.min()) if pitches.size else 0.0
    pitch_mean = float(np.mean(pitches)) if pitches.size else 0.0
    pitch_variance = float(np.var(pitches)) if pitches.size else 0.0

    durations = _note_durations(notes)
    note_duration_mean = float(np.mean(durations)) if durations.size else 0.0
    note_duration_variance = float(np.var(durations)) if durations.size else 0.0

    note_density = compute_note_density(midi)
    polyphony = compute_polyphony(midi)
    rhythmic_entropy = compute_rhythmic_entropy(notes)
    pitch_class_entropy = compute_pitch_class_entropy(notes)

    return {
        "duration_sec": duration_sec,
        "tempo_mean": tempo_mean,
        "tempo_std": tempo_std,
        "pitch_range": pitch_range,
        "pitch_mean": pitch_mean,
        "pitch_variance": pitch_variance,
        "note_duration_mean": note_duration_mean,
        "note_duration_variance": note_duration_variance,
        "note_density": note_density,
        "polyphony": polyphony,
        "rhythmic_entropy": rhythmic_entropy,
        "pitch_class_entropy": pitch_class_entropy,
    }

