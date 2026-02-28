"""
Sol-fa (Solfège) Phoneme System

Defines the phoneme vocabulary and mappings for singing voice synthesis.

Sol-fa syllables and their phoneme decomposition:
    Do  → D + OW   (like "dough")
    Re  → R + EY   (like "ray")
    Mi  → M + IY   (like "me")
    Fa  → F + AA   (like "far")
    Sol → S + OW   (like "so")
    La  → L + AA   (like "la")
    Ti  → T + IY   (like "tea")

MIDI pitch class mapping:
    0 (C)  → Do
    2 (D)  → Re
    4 (E)  → Mi
    5 (F)  → Fa
    7 (G)  → Sol
    9 (A)  → La
    11 (B) → Ti
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


# =============================================================================
# Phoneme Vocabulary
# =============================================================================

# Special tokens
PAD_TOKEN = '<pad>'
SIL_TOKEN = '<sil>'
UNK_TOKEN = '<unk>'

# Consonants (onset)
CONSONANTS = ['D', 'R', 'M', 'F', 'S', 'L', 'T']

# Vowels (nucleus) - using ARPAbet-style notation
VOWELS = ['OW', 'EY', 'IY', 'AA']

# Complete phoneme vocabulary
PHONEME_LIST = [
    PAD_TOKEN,   # 0: Padding
    SIL_TOKEN,   # 1: Silence
    UNK_TOKEN,   # 2: Unknown
    # Consonants
    'D',         # 3: Do onset
    'R',         # 4: Re onset
    'M',         # 5: Mi onset
    'F',         # 6: Fa onset
    'S',         # 7: Sol onset
    'L',         # 8: La onset
    'T',         # 9: Ti onset
    # Vowels
    'OW',        # 10: Do/Sol vowel (as in "go")
    'EY',        # 11: Re vowel (as in "say")
    'IY',        # 12: Mi/Ti vowel (as in "see")
    'AA',        # 13: Fa/La vowel (as in "father")
]

# Create vocabulary dictionaries
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_LIST)}
ID_TO_PHONEME = {i: p for i, p in enumerate(PHONEME_LIST)}
VOCAB_SIZE = len(PHONEME_LIST)

# Special token IDs
PAD_ID = PHONEME_TO_ID[PAD_TOKEN]
SIL_ID = PHONEME_TO_ID[SIL_TOKEN]
UNK_ID = PHONEME_TO_ID[UNK_TOKEN]


# =============================================================================
# Sol-fa Syllable Definitions
# =============================================================================

# Sol-fa syllables with phoneme decomposition
# Each syllable = (consonant, vowel)
SOLFA_SYLLABLES = {
    'Do':  ('D', 'OW'),
    'Re':  ('R', 'EY'),
    'Mi':  ('M', 'IY'),
    'Fa':  ('F', 'AA'),
    'Sol': ('S', 'OW'),
    'La':  ('L', 'AA'),
    'Ti':  ('T', 'IY'),
}

# MIDI pitch class to sol-fa syllable mapping
# Pitch class = midi_note % 12
PITCH_CLASS_TO_SOLFA = {
    0: 'Do',   # C
    2: 'Re',   # D
    4: 'Mi',   # E
    5: 'Fa',   # F
    7: 'Sol',  # G
    9: 'La',   # A
    11: 'Ti',  # B
}

# For chromatic notes (sharps/flats), map to nearest diatonic
# This handles edge cases in MIDI data
PITCH_CLASS_TO_SOLFA_CHROMATIC = {
    0: 'Do',   # C
    1: 'Do',   # C# -> Do (or Re)
    2: 'Re',   # D
    3: 'Re',   # D# -> Re (or Mi)
    4: 'Mi',   # E
    5: 'Fa',   # F
    6: 'Fa',   # F# -> Fa (or Sol)
    7: 'Sol',  # G
    8: 'Sol',  # G# -> Sol (or La)
    9: 'La',   # A
    10: 'La',  # A# -> La (or Ti)
    11: 'Ti',  # B
}


# =============================================================================
# Jianpu (Numbered Notation) Transformation
# =============================================================================

class Jianpu:
    """
    Jianpu (numbered notation) transformation for CCMZ dataset.

    Converts Western notation scale degrees to relative sol-fa syllables
    based on key signature. This is needed because CCMZ score.json uses
    absolute staff positions, but we need relative scale degrees for sol-fa.

    Reference: docs/CCMZ_TO_VPR_CONVERSION_GUIDE.md
    """

    # Transformation table for different key signatures
    # Index = fifths + 7 (to handle -7 to +7 range)
    jianpu_steps = [
        0, -4, -1, -5, -2, 1, -3,  # Flat keys: Cb(-7) to C(0)
        0,                          # C major (0)
        -4, -1, -5, -2, 1, -3, 0,  # Sharp keys: G(1) to C#(7)
    ]

    @classmethod
    def transform_step(cls, fifths: int, step: int) -> int:
        """
        Transform Western notation step to Jianpu scale degree.

        Args:
            fifths: Key signature (-7 to 7, where 0=C major, 1=G major, -1=F major)
            step: Original step from score.json (1-7, where 1=C, 2=D, 3=E, etc.)

        Returns:
            Transformed step representing relative scale degree (1-7)

        Example:
            In G major (fifths=1), step=2 (D on staff) becomes step=5 (Sol)
        """
        jiappu_step_index = step + 6
        jiappu_step_index += cls.jianpu_steps[fifths + 7]
        new_step = jiappu_step_index % 7 + 1
        return new_step


# Map scale degree (step) to sol-fa syllable
STEP_TO_SOLFA = {
    1: 'Do',   # 1st scale degree (tonic)
    2: 'Re',   # 2nd scale degree
    3: 'Mi',   # 3rd scale degree
    4: 'Fa',   # 4th scale degree
    5: 'Sol',  # 5th scale degree (dominant)
    6: 'La',   # 6th scale degree
    7: 'Ti',   # 7th scale degree (leading tone)
}


# =============================================================================
# Conversion Functions
# =============================================================================

def step_to_solfa(step: int) -> str:
    """
    Convert scale degree (step) to sol-fa syllable.

    This is the CORRECT way to determine sol-fa syllables from CCMZ data.
    Use this instead of midi_pitch_to_solfa() when score.json is available.

    Args:
        step: Scale degree (1-7)

    Returns:
        Sol-fa syllable name ('Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti')
    """
    return STEP_TO_SOLFA.get(step, 'Do')


def step_to_phonemes(step: int) -> Tuple[str, str]:
    """
    Convert scale degree to phonemes.

    Args:
        step: Scale degree (1-7)

    Returns:
        (consonant, vowel) phoneme tuple
    """
    solfa = step_to_solfa(step)
    return solfa_to_phonemes(solfa)


def midi_pitch_to_solfa(midi_pitch: int) -> str:
    """
    Convert MIDI pitch number to sol-fa syllable.

    WARNING: This is an APPROXIMATE mapping that assumes C major.
    For CCMZ dataset, use step_to_solfa() instead, which properly
    handles key signatures via Jianpu transformation.

    Args:
        midi_pitch: MIDI note number (0-127)

    Returns:
        Sol-fa syllable name ('Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti')
    """
    pitch_class = midi_pitch % 12
    return PITCH_CLASS_TO_SOLFA_CHROMATIC[pitch_class]


def solfa_to_phonemes(solfa: str) -> Tuple[str, str]:
    """
    Get phoneme decomposition for a sol-fa syllable.

    Args:
        solfa: Sol-fa syllable name

    Returns:
        (consonant, vowel) phoneme tuple
    """
    if solfa in SOLFA_SYLLABLES:
        return SOLFA_SYLLABLES[solfa]
    return (UNK_TOKEN, UNK_TOKEN)


def midi_pitch_to_phonemes(midi_pitch: int) -> Tuple[str, str]:
    """
    Convert MIDI pitch directly to phonemes.

    Args:
        midi_pitch: MIDI note number

    Returns:
        (consonant, vowel) phoneme tuple
    """
    solfa = midi_pitch_to_solfa(midi_pitch)
    return solfa_to_phonemes(solfa)


def phoneme_to_id(phoneme: str) -> int:
    """Convert phoneme string to ID."""
    return PHONEME_TO_ID.get(phoneme, UNK_ID)


def id_to_phoneme(phoneme_id: int) -> str:
    """Convert phoneme ID to string."""
    return ID_TO_PHONEME.get(phoneme_id, UNK_TOKEN)


# =============================================================================
# Frame-level Phoneme Generation
# =============================================================================

def generate_phoneme_labels(
    notes: List[Dict],
    tempo_map: List[Tuple[int, int]],
    measure0_tick: int,
    num_frames: int,
    frame_period_ms: float = 16.0,
    consonant_ratio: float = 0.30,
) -> np.ndarray:
    """
    Generate frame-level phoneme labels from MIDI notes.

    Each note is split into:
    - Consonant onset: first ~30% of note duration
    - Vowel nucleus: remaining ~70% of note duration

    Args:
        notes: List of note dicts with {tick, duration, pitch} or {tick, duration, step}
               If 'step' is present, uses step_to_phonemes() (CORRECT for CCMZ)
               If only 'pitch' is present, uses midi_pitch_to_phonemes() (fallback)
        tempo_map: List of (tick, tempo_us) tuples
        measure0_tick: Tick position of measure 0 (audio start)
        num_frames: Total number of frames
        frame_period_ms: Frame period in milliseconds
        consonant_ratio: Ratio of note duration for consonant (0.0-0.5)

    Returns:
        (num_frames,) array of phoneme IDs
    """
    from .world_processor import tick_to_ms

    # Initialize with silence
    phoneme_ids = np.full(num_frames, SIL_ID, dtype=np.int64)

    for note in notes:
        # Get note timing
        start_tick = note['tick']
        end_tick = start_tick + note['duration']

        # Convert to milliseconds
        start_ms = tick_to_ms(start_tick, tempo_map, measure0_tick)
        end_ms = tick_to_ms(end_tick, tempo_map, measure0_tick)

        # Convert to frames
        start_frame = int(start_ms / frame_period_ms)
        end_frame = int(end_ms / frame_period_ms)

        # Clip to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames))

        if start_frame >= end_frame:
            continue

        # Get phonemes for this note
        # Prefer 'step' (CORRECT) over 'pitch' (fallback)
        if 'step' in note:
            # Use step-based mapping (correct for CCMZ with score.json)
            consonant, vowel = step_to_phonemes(note['step'])
        else:
            # Fallback to pitch-based mapping (approximate, assumes C major)
            midi_pitch = note['pitch']
            consonant, vowel = midi_pitch_to_phonemes(midi_pitch)

        consonant_id = phoneme_to_id(consonant)
        vowel_id = phoneme_to_id(vowel)

        # Split note into consonant and vowel regions
        note_frames = end_frame - start_frame
        consonant_frames = max(1, int(note_frames * consonant_ratio))

        consonant_end = start_frame + consonant_frames

        # Assign phoneme IDs
        phoneme_ids[start_frame:consonant_end] = consonant_id
        phoneme_ids[consonant_end:end_frame] = vowel_id

    return phoneme_ids


def generate_phoneme_labels_simple(
    notes: List[Dict],
    tempo_map: List[Tuple[int, int]],
    measure0_tick: int,
    num_frames: int,
    frame_period_ms: float = 16.0,
) -> np.ndarray:
    """
    Generate frame-level phoneme labels (syllable-level, no C/V split).

    Uses a single ID per sol-fa syllable instead of splitting into C+V.
    This is simpler but loses the consonant/vowel distinction.

    Args:
        notes: List of note dicts with {tick, duration, pitch}
        tempo_map: List of (tick, tempo_us) tuples
        measure0_tick: Tick position of measure 0 (audio start)
        num_frames: Total number of frames
        frame_period_ms: Frame period in milliseconds

    Returns:
        (num_frames,) array of syllable IDs (0-7)
    """
    # Syllable-level vocabulary
    SYLLABLE_TO_ID = {
        '<sil>': 0,
        'Do': 1,
        'Re': 2,
        'Mi': 3,
        'Fa': 4,
        'Sol': 5,
        'La': 6,
        'Ti': 7,
    }

    from .world_processor import tick_to_ms

    # Initialize with silence
    syllable_ids = np.zeros(num_frames, dtype=np.int64)

    for note in notes:
        # Get note timing
        start_tick = note['tick']
        end_tick = start_tick + note['duration']

        # Convert to milliseconds
        start_ms = tick_to_ms(start_tick, tempo_map, measure0_tick)
        end_ms = tick_to_ms(end_tick, tempo_map, measure0_tick)

        # Convert to frames
        start_frame = int(start_ms / frame_period_ms)
        end_frame = int(end_ms / frame_period_ms)

        # Clip to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames))

        if start_frame >= end_frame:
            continue

        # Get sol-fa syllable
        midi_pitch = note['pitch']
        solfa = midi_pitch_to_solfa(midi_pitch)
        syllable_id = SYLLABLE_TO_ID.get(solfa, 0)

        # Assign to frames
        syllable_ids[start_frame:end_frame] = syllable_id

    return syllable_ids


# =============================================================================
# Utility Functions
# =============================================================================

def get_vocab_size() -> int:
    """Get the phoneme vocabulary size."""
    return VOCAB_SIZE


def get_syllable_vocab_size() -> int:
    """Get the syllable-level vocabulary size (simpler)."""
    return 8  # <sil> + 7 sol-fa syllables


def print_phoneme_info():
    """Print phoneme vocabulary information."""
    print("=" * 60)
    print("SOL-FA PHONEME SYSTEM")
    print("=" * 60)
    print()
    print("Vocabulary Size:", VOCAB_SIZE)
    print()
    print("Phoneme Vocabulary:")
    for i, p in enumerate(PHONEME_LIST):
        print(f"  {i:2d}: {p}")
    print()
    print("Sol-fa Syllables:")
    for solfa, (c, v) in SOLFA_SYLLABLES.items():
        print(f"  {solfa:3s} = {c} + {v}")
    print()
    print("MIDI Pitch Class Mapping:")
    for pc, solfa in PITCH_CLASS_TO_SOLFA.items():
        note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][pc]
        print(f"  {pc:2d} ({note_name:2s}) -> {solfa}")


if __name__ == '__main__':
    print_phoneme_info()

    # Test conversion
    print()
    print("Test MIDI pitch conversions:")
    for midi_note in [60, 62, 64, 65, 67, 69, 71, 72]:  # C4 to C5
        solfa = midi_pitch_to_solfa(midi_note)
        c, v = midi_pitch_to_phonemes(midi_note)
        print(f"  MIDI {midi_note} -> {solfa} -> {c} + {v}")
