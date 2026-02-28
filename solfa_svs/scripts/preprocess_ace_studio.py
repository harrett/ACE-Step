"""
Preprocess ACE Studio Exports with MIDI-Generated F0/Energy (Multi-Speaker)

Processes MP3 files exported from ACE Studio with corresponding MIDI files.
Supports multiple speakers with auto-assigned speaker IDs and incremental
processing. No WORLD vocoder — outputs only f0, energy, phonemes for the
DCAE latent pipeline.

- F0: Generated from MIDI pitch (not extracted from audio)
- Energy: Generated from MIDI velocity (not extracted from audio)
- Phonemes: Generated from MIDI lyrics

Usage:
    python -m solfa_svs.scripts.preprocess_ace_studio \
        --base_dir /data1/music/ACE-Step/data \
        --speakers ace_studio_exports Elirah Rowly \
        --midi_dir /data1/music/ACE-Step/data/midi_for_ace \
        --output_dir /data1/music/ACE-Step/data/processed

For testing with limited samples:
    python -m solfa_svs.scripts.preprocess_ace_studio \
        --base_dir /data1/music/ACE-Step/data \
        --speakers Elirah \
        --midi_dir /data1/music/ACE-Step/data/midi_for_ace \
        --output_dir /data1/music/ACE-Step/data/processed \
        --max_samples 5
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import mido
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from solfa_svs.data.solfa_phonemes import (
    generate_phoneme_labels,
    VOCAB_SIZE,
)


# Constants
SAMPLE_RATE = 44100
FRAME_PERIOD_MS = 11.6  # 512 / 44100 * 1000 ≈ 11.6ms (matches mel hop_length=512)

# Lyric to step mapping (sol-fa syllables)
# Chinese pinyin variants of sol-fa
LYRIC_TO_STEP = {
    # Standard sol-fa
    'Do': 1, 'Re': 2, 'Mi': 3, 'Fa': 4, 'Sol': 5, 'So': 5, 'La': 6, 'Ti': 7, 'Si': 7,
    # Chinese variants
    'Dao': 1, 'Duo': 1, 'Dou': 1,
    'Rei': 2, 'Rui': 2, 'Ruei': 2,
    'Xi': 7, 'Shi': 7,  # Xi/Si = Ti
    # Lowercase
    'do': 1, 're': 2, 'mi': 3, 'fa': 4, 'sol': 5, 'so': 5, 'la': 6, 'ti': 7, 'si': 7,
    'dao': 1, 'duo': 1, 'dou': 1,
    'rei': 2, 'rui': 2, 'ruei': 2,
    'xi': 7, 'shi': 7,
}


def parse_midi_file(midi_path: Path) -> Tuple[List[Dict], int, List[Tuple], int, Tuple[int, int]]:
    """
    Parse a standard MIDI file to extract notes with lyrics.

    Args:
        midi_path: Path to the MIDI file

    Returns:
        Tuple of (notes, ppqn, tempo_map, first_measure_ticks, time_sig)
        - notes: List of note dicts with {tick, duration, pitch, velocity, step}
        - ppqn: Pulses per quarter note
        - tempo_map: List of (tick, tempo_us) tuples
        - first_measure_ticks: Duration of first measure in ticks (for audio truncation)
        - time_sig: (beats, beat_unit) tuple
    """
    mid = mido.MidiFile(str(midi_path))
    ppqn = mid.ticks_per_beat

    # Build tempo map from all tracks and extract time signature
    tempo_map = []
    time_sig = (4, 4)  # Default 4/4

    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_time, msg.tempo))
            elif msg.type == 'time_signature':
                time_sig = (msg.numerator, msg.denominator)

    # Sort by tick and use first tempo if empty
    tempo_map.sort(key=lambda x: x[0])
    if not tempo_map:
        tempo_map = [(0, 500000)]  # Default 120 BPM

    # Calculate first measure duration in ticks
    # first_measure_ticks = ppqn * beats_per_bar
    beats, beat_unit = time_sig
    # For 4/4: ppqn * 4 = one bar. For 3/4: ppqn * 3, etc.
    first_measure_ticks = ppqn * beats * 4 // beat_unit

    # Find the track with notes (usually track 1)
    note_track = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on':
                note_track = track
                break
        if note_track:
            break

    if not note_track:
        return [], ppqn, tempo_map, first_measure_ticks, time_sig

    # Parse notes with lyrics
    notes = []
    current_time = 0
    active_notes = {}  # pitch -> (start_tick, velocity, lyric)
    current_lyric = None

    for msg in note_track:
        current_time += msg.time

        if msg.type == 'lyrics':
            current_lyric = msg.text
        elif msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = (current_time, msg.velocity, current_lyric)
            current_lyric = None  # Consume lyric
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start_tick, velocity, lyric = active_notes.pop(msg.note)
                duration = current_time - start_tick

                # Map lyric to step
                step = LYRIC_TO_STEP.get(lyric, 0) if lyric else 0

                notes.append({
                    'tick': start_tick,
                    'duration': duration,
                    'pitch': msg.note,
                    'velocity': velocity,
                    'step': step,
                    'lyric': lyric,
                })

    # Sort by tick
    notes.sort(key=lambda x: x['tick'])

    return notes, ppqn, tempo_map, first_measure_ticks, time_sig


def tick_to_ms(tick: int, tempo_map: List[Tuple], ppqn: int) -> float:
    """
    Convert MIDI ticks to milliseconds using tempo map.

    Args:
        tick: Tick position
        tempo_map: List of (tick, tempo_us) tuples
        ppqn: Pulses per quarter note

    Returns:
        Time in milliseconds
    """
    ms = 0.0
    prev_tick = 0
    tempo_us = tempo_map[0][1] if tempo_map else 500000

    for map_tick, new_tempo in tempo_map:
        if map_tick >= tick:
            break
        # Add time for this tempo segment
        delta_ticks = map_tick - prev_tick
        ms += (delta_ticks * tempo_us / 1000.0) / ppqn
        prev_tick = map_tick
        tempo_us = new_tempo

    # Add remaining ticks at current tempo
    remaining_ticks = tick - prev_tick
    ms += (remaining_ticks * tempo_us / 1000.0) / ppqn

    return ms


def generate_f0_from_midi(
    notes: List[Dict],
    tempo_map: List[Tuple],
    ppqn: int,
    num_frames: int,
    frame_period_ms: float = 16.0,
) -> np.ndarray:
    """
    Generate F0 contour from MIDI notes.

    Args:
        notes: List of note dicts with 'tick', 'duration', 'pitch'
        tempo_map: List of (tick, tempo_us) tuples
        ppqn: Pulses per quarter note
        num_frames: Number of frames to generate
        frame_period_ms: Frame period in milliseconds

    Returns:
        F0 contour (num_frames,) in Hz, 0 for unvoiced
    """
    f0 = np.zeros(num_frames, dtype=np.float32)

    for note in notes:
        start_tick = note['tick']
        end_tick = start_tick + note['duration']

        start_ms = tick_to_ms(start_tick, tempo_map, ppqn)
        end_ms = tick_to_ms(end_tick, tempo_map, ppqn)

        # Convert ms to frames
        start_frame = int(start_ms / frame_period_ms)
        end_frame = int(end_ms / frame_period_ms)

        # Clip to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        # Convert MIDI pitch to Hz: A4 (MIDI 69) = 440 Hz
        midi_pitch = note['pitch']
        f0_hz = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))

        # Assign F0 to frames
        for frame in range(start_frame, end_frame + 1):
            if frame < num_frames:
                f0[frame] = f0_hz

    # Apply light smoothing at transitions
    f0 = smooth_f0_transitions(f0, window=3)

    return f0


def smooth_f0_transitions(f0: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply light smoothing at note transitions."""
    f0_smooth = f0.copy()
    half_window = window // 2

    for i in range(half_window, len(f0) - half_window):
        if f0[i] > 0:
            if i > 0 and f0[i] != f0[i-1]:
                voiced_values = []
                for j in range(-half_window, half_window + 1):
                    if f0[i + j] > 0:
                        voiced_values.append(f0[i + j])
                if voiced_values:
                    f0_smooth[i] = np.mean(voiced_values)

    return f0_smooth


def generate_energy_from_midi(
    notes: List[Dict],
    tempo_map: List[Tuple],
    ppqn: int,
    num_frames: int,
    frame_period_ms: float = 16.0,
) -> np.ndarray:
    """
    Generate energy from MIDI velocity.

    Args:
        notes: List of note dicts with 'tick', 'duration', 'velocity'
        tempo_map: List of (tick, tempo_us) tuples
        ppqn: Pulses per quarter note
        num_frames: Number of frames to generate
        frame_period_ms: Frame period in milliseconds

    Returns:
        Energy contour (num_frames,)
    """
    energy = np.ones(num_frames, dtype=np.float32) * 0.5  # Default for silence

    for note in notes:
        start_tick = note['tick']
        end_tick = start_tick + note['duration']

        start_ms = tick_to_ms(start_tick, tempo_map, ppqn)
        end_ms = tick_to_ms(end_tick, tempo_map, ppqn)

        # Convert ms to frames
        start_frame = int(start_ms / frame_period_ms)
        end_frame = int(end_ms / frame_period_ms)

        # Clip to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        # Convert MIDI velocity (0-127) to energy
        # Use log scale for better distribution
        velocity = note.get('velocity', 64)
        note_energy = np.log(velocity / 127.0 + 0.1) * 3.0 + 1.0

        # Assign energy to frames
        for frame in range(start_frame, end_frame + 1):
            if frame < num_frames:
                energy[frame] = note_energy

    return energy


def load_or_create_speaker_map(speakers_json_path: Path) -> Dict[str, int]:
    """Load existing speaker map or create empty one."""
    if speakers_json_path.exists():
        with open(speakers_json_path) as f:
            return json.load(f)
    return {}


def save_speaker_map(speaker_map: Dict[str, int], speakers_json_path: Path):
    """Save speaker map to JSON."""
    speakers_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(speakers_json_path, 'w') as f:
        json.dump(speaker_map, f, indent=2)


def assign_speaker_ids(
    speaker_names: List[str],
    speaker_map: Dict[str, int],
) -> Dict[str, int]:
    """
    Assign speaker IDs, preserving existing assignments and adding new ones.

    New speakers get the next available ID (max existing + 1).
    """
    next_id = max(speaker_map.values()) + 1 if speaker_map else 0
    for name in speaker_names:
        if name not in speaker_map:
            speaker_map[name] = next_id
            next_id += 1
    return speaker_map


def is_train_sample(sample_id: str, train_ratio: float = 0.8) -> bool:
    """
    Deterministic hash-based train/val assignment.

    Ensures stable split even when new speakers are added later.
    """
    h = hashlib.md5(sample_id.encode()).hexdigest()
    return (int(h, 16) % 100) < int(train_ratio * 100)


def process_track(
    audio_path: Path,
    midi_path: Path,
    metadata_path: Path,
    output_dir: Path,
    speaker_name: str,
    speaker_id: int,
    min_duration: float = 10.0,
    max_duration: float = 300.0,
) -> Optional[Dict]:
    """
    Process a single ACE Studio track with MIDI-generated F0/energy.

    No WORLD vocoder — saves only f0, energy, phonemes for the DCAE
    latent pipeline.

    Args:
        audio_path: Path to the MP3 file
        midi_path: Path to the MIDI file
        metadata_path: Path to the JSON metadata file
        output_dir: Per-speaker output directory
        speaker_name: Name of the speaker
        speaker_id: Integer speaker ID
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds

    Returns:
        Metadata dict or None if processing failed
    """
    track_id = audio_path.stem

    # Parse MIDI file
    try:
        notes, ppqn, tempo_map, first_measure_ticks, time_sig = parse_midi_file(midi_path)
    except Exception as e:
        print(f"  SKIP {track_id}: Failed to parse MIDI: {e}")
        return None

    if not notes:
        print(f"  SKIP {track_id}: No notes in MIDI")
        return None

    # Load audio
    try:
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  SKIP {track_id}: Failed to load audio: {e}")
        return None

    duration = len(audio) / SAMPLE_RATE

    # Check duration
    if duration < min_duration:
        print(f"  SKIP {track_id}: Too short ({duration:.1f}s < {min_duration}s)")
        return None

    # Truncate if too long (instead of skipping to preserve limited data)
    truncated = False
    if duration > max_duration:
        print(f"  NOTE {track_id}: Truncating from {duration:.1f}s to {max_duration:.1f}s")
        max_samples = int(max_duration * SAMPLE_RATE)
        audio = audio[:max_samples]
        duration = max_duration
        truncated = True

    # Load metadata for additional info
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    # Normalize audio to consistent peak level
    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 0.9
        gain = target_peak / peak
        audio = audio * gain
        if gain > 2.0:
            print(f"  NOTE {track_id}: Audio normalized (gain={gain:.1f}x, peak was {peak:.3f})")

    # Calculate num_frames from audio duration (no WORLD needed)
    num_frames = int(np.floor(len(audio) / SAMPLE_RATE * 1000.0 / FRAME_PERIOD_MS)) + 1

    # Generate F0 from MIDI (NOT from audio!)
    f0 = generate_f0_from_midi(
        notes=notes,
        tempo_map=tempo_map,
        ppqn=ppqn,
        num_frames=num_frames,
        frame_period_ms=FRAME_PERIOD_MS,
    )

    # Generate energy from MIDI (NOT from audio!)
    energy = generate_energy_from_midi(
        notes=notes,
        tempo_map=tempo_map,
        ppqn=ppqn,
        num_frames=num_frames,
        frame_period_ms=FRAME_PERIOD_MS,
    )

    # Generate phoneme labels using step-based mapping
    STANDARD_PPQN = 480
    if ppqn != STANDARD_PPQN:
        scale = STANDARD_PPQN / ppqn
        normalized_notes = [
            {**note, 'tick': int(note['tick'] * scale), 'duration': int(note['duration'] * scale)}
            for note in notes
        ]
        normalized_tempo_map = [(int(tick * scale), tempo) for tick, tempo in tempo_map]
    else:
        normalized_notes = notes
        normalized_tempo_map = tempo_map

    try:
        phonemes = generate_phoneme_labels(
            notes=normalized_notes,
            tempo_map=normalized_tempo_map,
            measure0_tick=0,
            num_frames=num_frames,
            frame_period_ms=FRAME_PERIOD_MS,
            consonant_ratio=0.30,
        )
    except Exception as e:
        print(f"  SKIP {track_id}: Phoneme generation failed: {e}")
        return None

    # Count phoneme statistics
    unique_phonemes, counts = np.unique(phonemes, return_counts=True)
    phoneme_stats = {int(p): int(c) for p, c in zip(unique_phonemes, counts)}
    voiced_ratio = 1.0 - (phoneme_stats.get(1, 0) / num_frames)  # 1 = <sil>

    if voiced_ratio < 0.1:
        print(f"  SKIP {track_id}: Too few voiced frames ({voiced_ratio:.1%})")
        return None

    # Save processed data
    feature_filename = f"{track_id}.npz"
    audio_filename = f"{track_id}.wav"

    feature_path = output_dir / 'mel_features' / feature_filename
    audio_path_out = output_dir / 'audio' / audio_filename

    # Save features (f0, energy, phonemes only — no SP/AP)
    np.savez_compressed(
        feature_path,
        f0=f0.astype(np.float32),
        energy=energy.astype(np.float32),
        phonemes=phonemes.astype(np.int64),
    )

    # Save audio
    sf.write(audio_path_out, audio, SAMPLE_RATE)

    # Create metadata (paths relative to root output_dir)
    metadata = {
        'sample_id': track_id,
        'audio_path': f'{speaker_name}/audio/{audio_filename}',
        'feature_path': f'{speaker_name}/mel_features/{feature_filename}',
        'duration': duration,
        'num_frames': num_frames,
        'num_notes': len(notes),
        'voiced_ratio': voiced_ratio,
        'speaker_id': speaker_id,
        'speaker_name': speaker_name,
        'ppqn': ppqn,
        'tempo_bpm': meta.get('tempo_bpm', 120.0),
        'time_signature': f"{time_sig[0]}/{time_sig[1]}",
        'truncated': truncated,
    }

    return metadata


def process_speaker(
    speaker_name: str,
    speaker_id: int,
    base_dir: Path,
    midi_dir: Path,
    output_dir: Path,
    min_duration: float = 10.0,
    max_duration: float = 300.0,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Process all tracks for a single speaker with incremental skip logic.

    Returns list of metadata dicts for successfully processed tracks.
    """
    audio_dir = base_dir / speaker_name
    speaker_output_dir = output_dir / speaker_name

    # Create output directories
    (speaker_output_dir / 'mel_features').mkdir(parents=True, exist_ok=True)
    (speaker_output_dir / 'audio').mkdir(parents=True, exist_ok=True)

    # Find all MP3 files with matching MIDI
    mp3_files = sorted(audio_dir.glob('*.mp3'))
    if not mp3_files:
        print(f"  WARNING: No MP3 files found in {audio_dir}")
        return []

    print(f"  Found {len(mp3_files)} MP3 files")

    # Filter to those with matching MIDI
    valid_pairs = []
    for mp3_path in mp3_files:
        track_id = mp3_path.stem
        midi_path = midi_dir / f"{track_id}.mid"
        meta_path = midi_dir / f"{track_id}.json"

        if midi_path.exists():
            valid_pairs.append((mp3_path, midi_path, meta_path))

    if max_samples and len(valid_pairs) > max_samples:
        print(f"  Limiting to {max_samples} samples (from {len(valid_pairs)})")
        valid_pairs = valid_pairs[:max_samples]

    print(f"  {len(valid_pairs)} audio-MIDI pairs")

    if not valid_pairs:
        print(f"  WARNING: No valid audio-MIDI pairs for {speaker_name}")
        return []

    # Process files (with incremental skip)
    metadata_list = []
    skipped_existing = 0
    skipped_failed = 0

    for audio_path, midi_path, meta_path in tqdm(valid_pairs, desc=f"  {speaker_name}"):
        track_id = audio_path.stem

        # Incremental: skip if both output files already exist
        wav_out = speaker_output_dir / 'audio' / f"{track_id}.wav"
        npz_out = speaker_output_dir / 'mel_features' / f"{track_id}.npz"
        if wav_out.exists() and npz_out.exists():
            # Reload metadata from the existing NPZ to reconstruct metadata entry
            skipped_existing += 1
            continue

        result = process_track(
            audio_path=audio_path,
            midi_path=midi_path,
            metadata_path=meta_path,
            output_dir=speaker_output_dir,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        if result is not None:
            metadata_list.append(result)
        else:
            skipped_failed += 1

    # Also collect metadata for previously processed (skipped) tracks
    # by scanning existing output files
    if skipped_existing > 0:
        print(f"  Skipped {skipped_existing} already-processed tracks, rescanning metadata...")
        existing_npzs = sorted((speaker_output_dir / 'mel_features').glob('*.npz'))
        existing_ids = {p.stem for p in existing_npzs}
        newly_processed_ids = {m['sample_id'] for m in metadata_list}

        for npz_path in existing_npzs:
            track_id = npz_path.stem
            if track_id in newly_processed_ids:
                continue  # Already in metadata_list from this run
            wav_path = speaker_output_dir / 'audio' / f"{track_id}.wav"
            if not wav_path.exists():
                continue

            # Reconstruct metadata from saved files
            try:
                data = np.load(npz_path)
                f0 = data['f0']
                phonemes = data['phonemes']
                num_frames = len(f0)

                audio_info = sf.info(str(wav_path))
                duration = audio_info.duration

                unique_phonemes, counts = np.unique(phonemes, return_counts=True)
                phoneme_stats = {int(p): int(c) for p, c in zip(unique_phonemes, counts)}
                voiced_ratio = 1.0 - (phoneme_stats.get(1, 0) / num_frames) if num_frames > 0 else 0.0

                # Count notes from f0 (transitions from 0 to voiced)
                f0_bool = f0 > 0
                num_notes = int(np.sum(np.diff(f0_bool.astype(int)) == 1))

                metadata_list.append({
                    'sample_id': track_id,
                    'audio_path': f'{speaker_name}/audio/{track_id}.wav',
                    'feature_path': f'{speaker_name}/mel_features/{track_id}.npz',
                    'duration': duration,
                    'num_frames': num_frames,
                    'num_notes': num_notes,
                    'voiced_ratio': voiced_ratio,
                    'speaker_id': speaker_id,
                    'speaker_name': speaker_name,
                    'ppqn': 480,  # Default; exact value not critical for downstream
                    'tempo_bpm': 120.0,
                    'time_signature': '4/4',
                    'truncated': False,
                })
            except Exception as e:
                print(f"  WARNING: Could not reconstruct metadata for {track_id}: {e}")

    print(f"  Processed: {len(metadata_list)} total ({skipped_existing} existing, {skipped_failed} failed)")
    return metadata_list


def preprocess_ace_studio(
    base_dir: str,
    speakers: List[str],
    midi_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    min_duration: float = 10.0,
    max_duration: float = 300.0,
    max_samples: Optional[int] = None,
):
    """
    Preprocess ACE Studio exports for multiple speakers.

    Args:
        base_dir: Root directory containing speaker folders
        speakers: List of speaker directory names (relative to base_dir)
        midi_dir: Directory with shared MIDI files
        output_dir: Root output directory
        train_ratio: Ratio of samples for training (hash-based split)
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        max_samples: Maximum samples per speaker (None = all)
    """
    base_dir = Path(base_dir)
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create speaker map
    speakers_json_path = output_dir / 'speakers.json'
    speaker_map = load_or_create_speaker_map(speakers_json_path)
    speaker_map = assign_speaker_ids(speakers, speaker_map)
    save_speaker_map(speaker_map, speakers_json_path)

    print("=" * 80)
    print("PREPROCESSING ACE STUDIO EXPORTS (Multi-Speaker, MIDI F0/Energy)")
    print("=" * 80)
    print(f"Base dir: {base_dir}")
    print(f"MIDI dir: {midi_dir}")
    print(f"Output: {output_dir}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Frame period: {FRAME_PERIOD_MS} ms")
    print(f"Phoneme vocab size: {VOCAB_SIZE}")
    print(f"Train ratio: {train_ratio} (hash-based stable split)")
    if max_samples:
        print(f"Max samples per speaker: {max_samples} (TEST MODE)")
    print()
    print("Speakers:")
    for name in speakers:
        sid = speaker_map[name]
        print(f"  {name} -> speaker_id={sid}")
    print()

    # Process each speaker
    all_metadata = []
    for speaker_name in speakers:
        speaker_id = speaker_map[speaker_name]
        print(f"\n--- Processing speaker: {speaker_name} (id={speaker_id}) ---")

        speaker_metadata = process_speaker(
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            base_dir=base_dir,
            midi_dir=midi_dir,
            output_dir=output_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            max_samples=max_samples,
        )
        all_metadata.extend(speaker_metadata)

    if not all_metadata:
        print("\nERROR: No files processed across all speakers!")
        return

    # Hash-based stable train/val split
    train_metadata = []
    val_metadata = []
    for entry in all_metadata:
        if is_train_sample(entry['sample_id'], train_ratio):
            train_metadata.append(entry)
        else:
            val_metadata.append(entry)

    # Sort for deterministic output
    train_metadata.sort(key=lambda x: (x['speaker_id'], x['sample_id']))
    val_metadata.sort(key=lambda x: (x['speaker_id'], x['sample_id']))

    # Save combined metadata
    with open(output_dir / 'train_metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2)

    with open(output_dir / 'val_metadata.json', 'w') as f:
        json.dump(val_metadata, f, indent=2)

    # Print statistics
    total_duration = sum(m['duration'] for m in all_metadata)
    avg_duration = total_duration / len(all_metadata)
    avg_voiced = np.mean([m['voiced_ratio'] for m in all_metadata])

    print()
    print("=" * 80)
    print("PREPROCESSING COMPLETE (Multi-Speaker)")
    print("=" * 80)
    print(f"Total samples: {len(all_metadata)}")
    print(f"  Train: {len(train_metadata)}")
    print(f"  Val: {len(val_metadata)}")
    print(f"Total duration: {total_duration / 60:.2f} minutes ({total_duration / 3600:.2f} hours)")
    print(f"Average duration: {avg_duration:.1f} seconds")
    print(f"Average voiced ratio: {avg_voiced:.1%}")
    print()

    # Per-speaker stats
    print("Per-speaker breakdown:")
    for name in speakers:
        sid = speaker_map[name]
        spk_samples = [m for m in all_metadata if m['speaker_id'] == sid]
        spk_train = [m for m in train_metadata if m['speaker_id'] == sid]
        spk_val = [m for m in val_metadata if m['speaker_id'] == sid]
        spk_dur = sum(m['duration'] for m in spk_samples) if spk_samples else 0
        print(f"  {name} (id={sid}): {len(spk_samples)} samples "
              f"({len(spk_train)} train / {len(spk_val)} val), "
              f"{spk_dur / 60:.1f} min")

    print()
    print(f"Output files:")
    print(f"  {output_dir / 'speakers.json'}")
    print(f"  {output_dir / 'train_metadata.json'}")
    print(f"  {output_dir / 'val_metadata.json'}")
    for name in speakers:
        print(f"  {output_dir / name / 'audio/'}")
        print(f"  {output_dir / name / 'mel_features/'}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess ACE Studio exports (multi-speaker, MIDI F0/energy)'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Root directory containing speaker folders and MIDI'
    )
    parser.add_argument(
        '--speakers',
        type=str,
        nargs='+',
        required=True,
        help='Speaker directory names (relative to --base_dir)'
    )
    parser.add_argument(
        '--midi_dir',
        type=str,
        required=True,
        help='Directory with shared MIDI files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Root output directory for processed data'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of samples for training (default: 0.8)'
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=10.0,
        help='Minimum audio duration in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=300.0,
        help='Maximum audio duration in seconds (default: 300.0). Longer clips are truncated.'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples per speaker (None = all, for testing)'
    )

    args = parser.parse_args()

    preprocess_ace_studio(
        base_dir=args.base_dir,
        speakers=args.speakers,
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()
