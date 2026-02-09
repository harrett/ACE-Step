"""
MIDI Parser: Convert score/MIDI input to note events + frame-level features.

Two modes:
1. From MIDI (production): .mid files with sol-fa lyric events → synthesize features
2. From NPZ (evaluation): load extracted features and resample

The .mid files from midi_for_ace/ contain embedded sol-fa lyric events
(Chinese pinyin: Dao, Rei, Mi, Fa, So, La, Xi) paired with MIDI notes.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional


# =============================================================================
# Phoneme Vocabulary (consonant + vowel decomposition)
# =============================================================================

# Special tokens
PAD_TOKEN = '<pad>'
SIL_TOKEN = '<sil>'
UNK_TOKEN = '<unk>'

# Complete phoneme vocabulary (14 tokens)
PHONEME_LIST = [
    PAD_TOKEN,   # 0: Padding
    SIL_TOKEN,   # 1: Silence
    UNK_TOKEN,   # 2: Unknown
    # Consonants (onset)
    'D',         # 3: Do onset
    'R',         # 4: Re onset
    'M',         # 5: Mi onset
    'F',         # 6: Fa onset
    'S',         # 7: Sol onset
    'L',         # 8: La onset
    'T',         # 9: Ti onset
    # Vowels (nucleus)
    'OW',        # 10: Do/Sol vowel (as in "go")
    'EY',        # 11: Re vowel (as in "say")
    'IY',        # 12: Mi/Ti vowel (as in "see")
    'AA',        # 13: Fa/La vowel (as in "father")
]

PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_LIST)}
ID_TO_PHONEME = {i: p for i, p in enumerate(PHONEME_LIST)}
NUM_PHONEMES = len(PHONEME_LIST)  # 14

PAD_ID = 0
SIL_ID = 1
UNK_ID = 2

# Sol-fa syllable decomposition: syllable → (consonant, vowel)
SOLFA_SYLLABLES = {
    'Do':  ('D', 'OW'),
    'Re':  ('R', 'EY'),
    'Mi':  ('M', 'IY'),
    'Fa':  ('F', 'AA'),
    'Sol': ('S', 'OW'),
    'La':  ('L', 'AA'),
    'Ti':  ('T', 'IY'),
}

# Chinese pinyin sol-fa names used in ACE Studio MIDI lyric events
PINYIN_TO_SOLFA = {
    'Dao': 'Do',
    'dao': 'Do',
    'Rei': 'Re',
    'rei': 'Re',
    'Mi':  'Mi',
    'mi':  'Mi',
    'Fa':  'Fa',
    'fa':  'Fa',
    'So':  'Sol',
    'so':  'Sol',
    'La':  'La',
    'la':  'La',
    'Xi':  'Ti',
    'xi':  'Ti',
    # Also accept standard western names
    'Do':  'Do',
    'do':  'Do',
    'Re':  'Re',
    're':  'Re',
    'Sol': 'Sol',
    'sol': 'Sol',
    'Ti':  'Ti',
    'ti':  'Ti',
    'Si':  'Ti',
    'si':  'Ti',
}

# MIDI pitch class → sol-fa (chromatic, assumes C major; fallback when no lyrics)
PITCH_CLASS_TO_SOLFA = {
    0: 'Do',   1: 'Do',   2: 'Re',   3: 'Re',
    4: 'Mi',   5: 'Fa',   6: 'Fa',   7: 'Sol',
    8: 'Sol',  9: 'La',  10: 'La',  11: 'Ti',
}


# =============================================================================
# Conversion Helpers
# =============================================================================

def midi_pitch_to_hz(midi_pitch: int) -> float:
    """Convert MIDI note number to Hz. Returns 0 for pitch 0 (rest)."""
    if midi_pitch <= 0:
        return 0.0
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    """Convert F0 in Hz to MIDI note number. Unvoiced (f0<=0) maps to 0."""
    midi = np.zeros_like(f0_hz)
    voiced = f0_hz > 0
    midi[voiced] = 69 + 12 * np.log2(f0_hz[voiced] / 440.0 + 1e-12)
    return midi


def solfa_to_phoneme_ids(solfa: str) -> Tuple[int, int]:
    """
    Get (consonant_id, vowel_id) for a sol-fa syllable name.

    Args:
        solfa: Sol-fa syllable ('Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti')

    Returns:
        (consonant_id, vowel_id) tuple of phoneme IDs
    """
    if solfa in SOLFA_SYLLABLES:
        c, v = SOLFA_SYLLABLES[solfa]
        return PHONEME_TO_ID[c], PHONEME_TO_ID[v]
    return UNK_ID, UNK_ID


def lyric_to_solfa(lyric: str) -> Optional[str]:
    """
    Map a MIDI lyric event text to a sol-fa syllable name.

    Handles both Chinese pinyin (Dao, Rei, So, Xi) and Western names.
    Returns None if unrecognized.
    """
    lyric = lyric.strip()
    return PINYIN_TO_SOLFA.get(lyric)


def pitch_to_solfa(midi_pitch: int) -> str:
    """Fallback: map MIDI pitch to sol-fa syllable (assumes C major)."""
    return PITCH_CLASS_TO_SOLFA[midi_pitch % 12]


# =============================================================================
# Feature Synthesis from Note Events
# =============================================================================

def synthesize_features_from_notes(
    notes: List[Dict],
    duration_sec: float,
    frame_rate: float = 10.766601562,
    consonant_ratio: float = 0.30,
    energy_scale: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Synthesize frame-level F0, energy, and phonemes from note events.

    This is the PRODUCTION path — only the score (note events) is needed.
    Each note is split into consonant onset (~30%) and vowel nucleus (~70%).
    F0 is a flat pitch per note (expression added later).
    Energy is derived from velocity, scaled to match training audio RMS range.

    Args:
        notes: List of note event dicts. Each must have:
            - solfa (str): sol-fa syllable name ('Do', 'Re', etc.)
            - midi_pitch (int): MIDI note number (0 = rest)
            - onset_sec (float): note onset in seconds
            - offset_sec (float): note offset in seconds
            - velocity (int, optional): 0-127, default 80
        duration_sec: total duration in seconds
        frame_rate: target frame rate (DCAE latent rate)
        consonant_ratio: fraction of note for consonant onset (default 0.30)
        energy_scale: scale factor mapping MIDI velocity to training energy range.
            Training energy is mel RMS (~0.11 mean at 44.1kHz), not raw velocity/127 (~0.4).
            Default 0.35 calibrated as training_mean / raw_inference_mean = 0.111 / 0.315.

    Returns:
        f0: (L,) F0 in Hz at frame rate
        energy: (L,) energy at frame rate
        phonemes: (L,) phoneme IDs at frame rate
        notes_enriched: note events matching training format (C/V segments)
    """
    L = int(np.ceil(duration_sec * frame_rate))

    f0 = np.zeros(L, dtype=np.float32)
    energy = np.zeros(L, dtype=np.float32)
    phonemes = np.full(L, SIL_ID, dtype=np.int64)

    for note in notes:
        # Determine sol-fa syllable
        solfa = note.get("solfa")
        if solfa is None:
            midi_pitch = int(note.get("midi_pitch", 0))
            if midi_pitch > 0:
                solfa = pitch_to_solfa(midi_pitch)

        midi_pitch = int(note.get("midi_pitch", 0))
        onset_sec = float(note["onset_sec"])
        offset_sec = float(note["offset_sec"])
        velocity = int(note.get("velocity", 80))

        onset_frame = int(np.round(onset_sec * frame_rate))
        offset_frame = int(np.round(offset_sec * frame_rate))
        onset_frame = max(0, min(onset_frame, L))
        offset_frame = max(onset_frame, min(offset_frame, L))

        if onset_frame >= offset_frame:
            continue

        # F0 from MIDI pitch
        f0[onset_frame:offset_frame] = midi_pitch_to_hz(midi_pitch)

        # Energy from velocity, scaled to match training audio RMS range,
        # with within-note envelope to match training energy dynamics.
        # Training audio has natural attack/decay variation (std≈0.036);
        # flat energy (std≈0.013) produces lower-quality output.
        base_energy = (velocity / 127.0) * energy_scale
        note_frames = offset_frame - onset_frame
        if note_frames > 1:
            t_norm = np.linspace(0, 1, note_frames)
            # Attack-sustain-decay envelope:
            #   attack (0-15%): ramp 0.4 -> 1.1 (quick onset with slight overshoot)
            #   sustain (15-80%): settle 1.1 -> 0.9 (gradual natural decay)
            #   release (80-100%): decay 0.9 -> 0.3
            envelope = np.where(
                t_norm < 0.15,
                0.4 + (1.1 - 0.4) * (t_norm / 0.15),
                np.where(
                    t_norm < 0.80,
                    1.1 - (1.1 - 0.9) * ((t_norm - 0.15) / 0.65),
                    0.9 - (0.9 - 0.3) * ((t_norm - 0.80) / 0.20),
                ),
            )
            energy[onset_frame:offset_frame] = base_energy * envelope
        else:
            energy[onset_frame:offset_frame] = base_energy

        # Phonemes: consonant + vowel split
        if solfa is not None and solfa in SOLFA_SYLLABLES:
            c_id, v_id = solfa_to_phoneme_ids(solfa)
            note_frames = offset_frame - onset_frame
            consonant_frames = max(1, int(note_frames * consonant_ratio))
            consonant_end = onset_frame + consonant_frames

            phonemes[onset_frame:consonant_end] = c_id
            phonemes[consonant_end:offset_frame] = v_id

    # Extract note events from synthesized frame-level features.
    # This matches the training format where extract_note_events() segments
    # by phoneme changes, producing separate entries for consonant and vowel.
    notes_enriched = extract_note_events(phonemes, f0, energy, frame_rate)

    return f0, energy, phonemes, notes_enriched


# =============================================================================
# MIDI File Loading (Production Path)
# =============================================================================

def load_midi_file(
    midi_path: str,
    metadata_path: Optional[str] = None,
    frame_rate: float = 10.766601562,
    energy_scale: float = 0.35,
) -> Dict:
    """
    Load a MIDI file with sol-fa lyric events and synthesize frame-level features.

    Expects MIDI files from midi_for_ace/ format: each note is preceded by a
    lyric event (0xFF 0x05) containing the Chinese pinyin sol-fa syllable
    (Dao, Rei, Mi, Fa, So, La, Xi).

    Args:
        midi_path: path to .mid file
        metadata_path: optional path to companion .json metadata file
            (provides key signature, tempo, etc.)
        frame_rate: target frame rate
        energy_scale: scale factor for MIDI velocity → energy mapping

    Returns:
        Dict with f0, energy, phonemes, notes, duration_sec
    """
    import mido

    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    duration_sec = mid.length

    # Build tempo map from all tracks: list of (abs_tick, tempo_us)
    tempo_changes = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_changes.append((abs_tick, msg.tempo))
    if not tempo_changes:
        tempo_changes = [(0, 500000)]  # default 120 BPM
    tempo_changes.sort(key=lambda x: x[0])

    # Build cumulative time map: (tick, cumulative_seconds, tempo)
    time_map = [(0, 0.0, tempo_changes[0][1])]
    for i in range(1, len(tempo_changes)):
        prev_tick, prev_sec, prev_tempo = time_map[-1]
        cur_tick, cur_tempo = tempo_changes[i]
        delta_sec = mido.tick2second(cur_tick - prev_tick, tpb, prev_tempo)
        time_map.append((cur_tick, prev_sec + delta_sec, cur_tempo))

    def tick_to_sec(tick):
        """Convert absolute tick to seconds using tempo map."""
        # Find the applicable tempo segment (last entry with tick <= target)
        for i in range(len(time_map) - 1, -1, -1):
            base_tick, base_sec, tempo = time_map[i]
            if tick >= base_tick:
                return base_sec + mido.tick2second(tick - base_tick, tpb, tempo)
        return 0.0

    # Merge tracks and iterate in ticks, converting to seconds
    notes = []
    pending_lyric = None
    active_notes = {}  # pitch -> (onset_tick, velocity, lyric)

    current_tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        current_tick += msg.time

        if msg.type == "lyrics":
            pending_lyric = msg.text

        elif msg.type == "text":
            pending_lyric = msg.text

        elif msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = (current_tick, msg.velocity, pending_lyric)
            pending_lyric = None

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active_notes:
                onset_tick, velocity, lyric = active_notes.pop(msg.note)
                onset_sec = tick_to_sec(onset_tick)
                offset_sec = tick_to_sec(current_tick)

                # Determine sol-fa syllable from lyric event
                solfa = None
                if lyric is not None:
                    solfa = lyric_to_solfa(lyric)

                # Fallback: derive from MIDI pitch (assumes C major)
                if solfa is None:
                    solfa = pitch_to_solfa(msg.note)

                notes.append({
                    "solfa": solfa,
                    "midi_pitch": msg.note,
                    "onset_sec": onset_sec,
                    "offset_sec": offset_sec,
                    "velocity": velocity,
                })

    # Sort by onset
    notes.sort(key=lambda n: n["onset_sec"])

    f0, energy, phonemes, notes_enriched = synthesize_features_from_notes(
        notes, duration_sec, frame_rate, energy_scale=energy_scale
    )

    return {
        "f0": f0,
        "energy": energy,
        "phonemes": phonemes,
        "notes": notes_enriched,
        "duration_sec": duration_sec,
    }


# =============================================================================
# NPZ Loading (Evaluation Path)
# =============================================================================

def extract_note_events(
    phonemes: np.ndarray,
    f0: np.ndarray,
    energy: np.ndarray,
    frame_rate: float,
) -> List[Dict]:
    """
    Extract note-level events from frame-level phoneme + f0 + energy sequences.

    Each note event corresponds to a contiguous segment of the same phoneme
    with voiced F0 (or a silence/breath segment).

    Args:
        phonemes: (L,) phoneme IDs at frame rate
        f0: (L,) F0 in Hz at frame rate
        energy: (L,) energy at frame rate
        frame_rate: frames per second

    Returns:
        List of note event dicts with keys:
            onset_frame, offset_frame, onset_sec, offset_sec,
            midi_pitch, velocity, phoneme_id, duration_sec
    """
    L = len(phonemes)
    notes = []

    if L == 0:
        return notes

    # Segment by phoneme changes
    seg_start = 0
    for i in range(1, L + 1):
        if i == L or phonemes[i] != phonemes[seg_start]:
            seg_end = i
            phoneme_id = int(phonemes[seg_start])

            # Compute note properties from this segment
            onset_sec = seg_start / frame_rate
            offset_sec = seg_end / frame_rate
            duration_sec = offset_sec - onset_sec

            seg_f0 = f0[seg_start:seg_end]
            seg_energy = energy[seg_start:seg_end]

            # MIDI pitch: median of voiced F0 in segment
            voiced_f0 = seg_f0[seg_f0 > 0]
            if len(voiced_f0) > 0:
                median_hz = np.median(voiced_f0)
                midi_pitch = int(np.round(69 + 12 * np.log2(median_hz / 440.0 + 1e-12)))
                midi_pitch = np.clip(midi_pitch, 0, 127)
            else:
                midi_pitch = 0  # rest / silence

            # Velocity from energy (quantize to 0-127)
            mean_energy = np.mean(seg_energy) if len(seg_energy) > 0 else 0.0
            velocity = int(np.clip(mean_energy * 127, 0, 127))

            notes.append({
                "onset_frame": seg_start,
                "offset_frame": seg_end,
                "onset_sec": float(onset_sec),
                "offset_sec": float(offset_sec),
                "duration_sec": float(duration_sec),
                "midi_pitch": int(midi_pitch),
                "velocity": int(velocity),
                "phoneme_id": int(phoneme_id),
            })

            seg_start = i

    return notes


def resample_features(
    f0: np.ndarray,
    energy: np.ndarray,
    phonemes: np.ndarray,
    source_rate: float,
    target_rate: float,
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample frame-level features from source frame rate to target frame rate.

    Uses linear interpolation for continuous features (f0, energy)
    and nearest-neighbor for discrete features (phonemes).

    Args:
        f0: (T_src,) F0 in Hz
        energy: (T_src,) energy
        phonemes: (T_src,) phoneme IDs
        source_rate: source frame rate (e.g., 86.13 fps)
        target_rate: target frame rate (e.g., 10.77 fps)
        target_length: optional explicit target length

    Returns:
        f0_resampled, energy_resampled, phonemes_resampled - all (T_tgt,)
    """
    T_src = len(f0)
    if target_length is None:
        target_length = int(np.round(T_src * target_rate / source_rate))

    if target_length == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)

    # Source and target time indices
    src_times = np.arange(T_src) / source_rate
    tgt_times = np.arange(target_length) / target_rate

    # Linear interpolation for continuous features
    f0_resampled = np.interp(tgt_times, src_times, f0)
    energy_resampled = np.interp(tgt_times, src_times, energy)

    # Nearest-neighbor for phonemes
    src_indices = np.clip(
        np.round(tgt_times * source_rate).astype(int),
        0,
        T_src - 1,
    )
    phonemes_resampled = phonemes[src_indices]

    return f0_resampled, energy_resampled, phonemes_resampled


def load_features_from_npz(
    npz_path: str,
    source_frame_rate: float = 44100 / 512,
    target_frame_rate: float = 10.766601562,
    target_length: Optional[int] = None,
) -> Dict:
    """
    Load features from an NPZ file and resample to latent frame rate.

    Args:
        npz_path: path to .npz file with keys: f0, phonemes, energy
        source_frame_rate: frame rate of the NPZ features (86.13 for hop=512 at 44.1kHz)
        target_frame_rate: target frame rate (DCAE latent rate)
        target_length: explicit target length (from DCAE encoding)

    Returns:
        Dict with resampled f0, energy, phonemes, and extracted note events
    """
    data = np.load(npz_path)
    f0 = data["f0"].astype(np.float32)
    phonemes = data["phonemes"].astype(np.int64)
    energy = data["energy"].astype(np.float32)

    f0_rs, energy_rs, phonemes_rs = resample_features(
        f0, energy, phonemes,
        source_rate=source_frame_rate,
        target_rate=target_frame_rate,
        target_length=target_length,
    )

    notes = extract_note_events(phonemes_rs, f0_rs, energy_rs, target_frame_rate)

    return {
        "f0": f0_rs.astype(np.float32),
        "energy": energy_rs.astype(np.float32),
        "phonemes": phonemes_rs.astype(np.int64),
        "notes": notes,
    }
