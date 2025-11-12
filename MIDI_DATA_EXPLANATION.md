# Understanding MIDI Data Structure

## Overview

The dataset contains **file paths** to MIDI files. Each MIDI file contains structured musical data that represents a piece of music in a digital format.

## Dataset Structure

### What's in the Dataset
- **Field**: `file_name` (string)
- **Value**: Path to the MIDI file (e.g., `"data/albeniz-espana_op_165.mid"`)
- **Total examples**: ~4,800 MIDI files

The dataset itself is just a catalog of file paths. The actual musical data is stored in the `.mid` files themselves.

---

## MIDI File Structure

When you parse a MIDI file, it contains the following components:

### 1. **File-Level Metadata**

```
Type: 1
Ticks per beat: 480
Length (seconds): 95.19
Number of tracks: 9
```

- **Type**: MIDI file format type
  - Type 0: Single track (all data in one track)
  - Type 1: Multiple tracks (synchronized, like a multi-track recording)
  - Type 2: Multiple songs (independent tracks)
  
- **Ticks per beat**: The resolution/timing precision
  - Higher = more precise timing
  - 480 ticks per beat means each beat is divided into 480 parts
  - Used to calculate exact note durations

- **Length**: Total duration of the piece in seconds

- **Number of tracks**: How many separate musical parts (like instruments or voices)

### 2. **Tracks**

Each track represents a separate musical part (like Piano Right, Piano Left, etc.):

```
Track 0: 'Espana Op. 165' (568 messages)
Track 1: 'Piano right' (739 messages)
Track 2: 'Piano left' (709 messages)
```

- **Track name**: Descriptive name (often the instrument or part name)
- **Messages**: All the MIDI events in that track (notes, tempo changes, etc.)

### 3. **MIDI Messages**

MIDI files contain different types of messages:

#### **Note Messages**
- **`note_on`**: Start playing a note
  - `note`: MIDI note number (0-127, where 60 = middle C)
  - `velocity`: How hard the note is struck (0-127, affects volume)
  - `time`: Time delay before this event (in ticks)
  
- **`note_off`**: Stop playing a note
  - Same parameters as note_on
  - Velocity of 0 in note_on also means note off

#### **Tempo Messages**
- **`set_tempo`**: Change the playback speed
  - `tempo`: Microseconds per quarter note
  - Lower tempo = faster music
  - Example: `tempo=642192` means 642,192 microseconds per beat
  - This converts to approximately 93.4 BPM (beats per minute)

#### **Program Change Messages**
- **`program_change`**: Change the instrument/sound
  - `program`: Instrument number (0-127)
  - Standard MIDI instrument numbers (0=Piano, 40=Violin, etc.)

#### **Meta Messages**
- **`set_tempo`**: Tempo changes (shown in your output)
- **`track_name`**: Track names
- **`copyright`**: Copyright information
- **`text`**: Text annotations

### 4. **Note Data Example**

```
Note: A5 (MIDI 81), Velocity: 60, Time: 240
Note: E6 (MIDI 88), Velocity: 66, Time: 0
```

Breaking this down:

- **Note Name**: `A5` = A note in the 5th octave
  - MIDI note numbers: 0 = C-1, 60 = C4 (middle C), 127 = G9
  - Formula: `note_name = (note_number // 12) - 1` for octave, `note_number % 12` for pitch class
  
- **MIDI Number**: `81` = the raw MIDI note value
  - Each semitone = +1 MIDI number
  - C4 = 60, C#4 = 61, D4 = 62, etc.

- **Velocity**: `60` = how hard the key is pressed (0-127)
  - 0 = silent
  - 64 = medium
  - 127 = maximum
  - Affects volume and sometimes timbre

- **Time**: `240` = delay in ticks before this note plays
  - 0 = plays immediately after previous event
  - 240 ticks = if ticks_per_beat is 480, this is half a beat later
  - Time is cumulative/delta time between events

---

## How MIDI Represents Music

### Musical Information Stored:

1. **Pitch**: Which notes to play (MIDI note numbers)
2. **Timing**: When to play them (time deltas in ticks)
3. **Duration**: How long notes last (from note_on to note_off)
4. **Volume**: How loud (velocity)
5. **Tempo**: Speed of playback (tempo messages)
6. **Instruments**: What sounds to use (program_change)
7. **Structure**: Multiple tracks for different parts

### What MIDI Does NOT Store:

- **Audio waveforms**: MIDI is not audio, it's instructions
- **Exact sound quality**: Depends on the synthesizer/player
- **Visual notation**: No sheet music (but can be converted)

---

## Data Flow Example

Let's trace what happens when you play a MIDI file:

1. **File is loaded**: Parser reads the MIDI file structure
2. **Tracks are identified**: Each track represents a musical part
3. **Messages are processed in order**:
   - Track 0: Sets tempo to 642192 microseconds/beat
   - Track 1: After 240 ticks, play A5 with velocity 60
   - Track 1: Immediately play E6 with velocity 66
   - Track 1: Immediately play D6 with velocity 55
   - ... and so on
4. **Synthesizer converts to sound**: A MIDI player interprets these messages and generates audio

---

## Key Concepts

### Delta Time
- Time values are **relative** (delta time), not absolute
- Each message's time is the delay since the previous message
- This makes editing easier (inserting events doesn't require renumbering)

### Tracks vs. Channels
- **Tracks**: Organizational structure (like separate parts)
- **Channels**: MIDI channels (0-15) for routing to different instruments
- A track can contain messages for multiple channels

### Note On/Off
- A note starts with `note_on` (velocity > 0)
- A note ends with `note_off` OR `note_on` with velocity = 0
- Duration = time between note_on and note_off

---

## Summary

**Dataset Level:**
- Just file paths (strings)

**MIDI File Level:**
- File metadata (type, ticks per beat, length, track count)
- Multiple tracks (each with a name and messages)
- Messages (notes, tempo, instrument changes, etc.)

**Note Level:**
- Pitch (MIDI number → note name)
- Timing (delta time in ticks)
- Velocity (0-127)
- Duration (from note_on to note_off)

This structure allows MIDI to represent complex musical pieces in a compact, editable format that can be played back by any MIDI-compatible device or software.

