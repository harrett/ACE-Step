"""
F0 Expression Generator: Vibrato and Portamento

Transforms flat MIDI-derived F0 into expressive F0 with natural singing characteristics.

Vibrato: Sinusoidal pitch modulation (5-6 Hz, 2-4% depth)
Portamento: Smooth pitch transitions between notes
"""

import numpy as np
from typing import Optional, Tuple


def add_vibrato(
    f0: np.ndarray,
    frame_period_ms: float = 16.0,
    vibrato_rate_hz: float = 5.5,
    vibrato_depth: float = 0.03,
    min_note_frames: int = 18,
    onset_delay_frames: int = 9,
    ramp_in_frames: int = 6,
) -> np.ndarray:
    """
    Add vibrato to F0 contour.

    Vibrato is applied to sustained notes (longer than min_note_frames).
    The vibrato starts after onset_delay_frames and ramps in over ramp_in_frames.

    Args:
        f0: (T,) F0 in Hz. 0 values indicate unvoiced frames.
        frame_period_ms: Frame period in milliseconds (default 16ms)
        vibrato_rate_hz: Vibrato frequency in Hz (natural range: 5-6 Hz)
        vibrato_depth: Vibrato depth as fraction of F0 (0.03 = 3%)
        min_note_frames: Minimum note length to apply vibrato (18 frames = ~288ms)
        onset_delay_frames: Frames to wait before starting vibrato (9 = ~144ms)
        ramp_in_frames: Frames over which vibrato depth ramps up (6 = ~96ms)

    Returns:
        f0_vibrato: (T,) F0 with vibrato applied
    """
    f0_vibrato = f0.copy()
    T = len(f0)

    # Find voiced segments (continuous regions with f0 > 0)
    voiced_mask = f0 > 0

    # Identify note boundaries (transitions in/out of voiced)
    segment_starts = []
    segment_ends = []

    in_segment = False
    for i in range(T):
        if voiced_mask[i] and not in_segment:
            segment_starts.append(i)
            in_segment = True
        elif not voiced_mask[i] and in_segment:
            segment_ends.append(i)
            in_segment = False

    # Close final segment
    if in_segment:
        segment_ends.append(T)

    # Apply vibrato to each voiced segment
    frame_period_s = frame_period_ms / 1000.0

    for start, end in zip(segment_starts, segment_ends):
        note_length = end - start

        # Only apply vibrato to notes longer than threshold
        if note_length < min_note_frames:
            continue

        # Time array for this segment
        t = np.arange(note_length) * frame_period_s

        # Sinusoidal vibrato
        vibrato = np.sin(2 * np.pi * vibrato_rate_hz * t)

        # Create depth envelope: 0 during onset delay, then ramp up
        depth_envelope = np.zeros(note_length)

        for i in range(note_length):
            if i < onset_delay_frames:
                # No vibrato during onset
                depth_envelope[i] = 0.0
            elif i < onset_delay_frames + ramp_in_frames:
                # Ramp in
                ramp_progress = (i - onset_delay_frames) / ramp_in_frames
                depth_envelope[i] = vibrato_depth * ramp_progress
            else:
                # Full vibrato
                depth_envelope[i] = vibrato_depth

        # Apply vibrato: f0 * (1 + depth * sin(...))
        f0_vibrato[start:end] = f0[start:end] * (1 + depth_envelope * vibrato)

    return f0_vibrato


def add_portamento(
    f0: np.ndarray,
    phonemes: Optional[np.ndarray] = None,
    transition_frames: int = 3,
) -> np.ndarray:
    """
    Add portamento (pitch glides) between consecutive voiced notes.

    Smooths pitch transitions at note boundaries using linear interpolation.

    Args:
        f0: (T,) F0 in Hz. 0 values indicate unvoiced frames.
        phonemes: (T,) optional phoneme IDs. If provided, transitions only occur
                  at phoneme boundaries within voiced regions.
        transition_frames: Number of frames for glide (3 frames = ~48ms)

    Returns:
        f0_portamento: (T,) F0 with portamento applied
    """
    f0_porta = f0.copy()
    T = len(f0)

    if phonemes is not None:
        # Find phoneme boundaries within voiced regions
        voiced_mask = f0 > 0

        for i in range(1, T - transition_frames):
            # Check for phoneme boundary
            if phonemes[i] != phonemes[i - 1]:
                # Check if both sides are voiced
                if voiced_mask[i - 1] and voiced_mask[i]:
                    # Get pitch values before and after boundary
                    pitch_before = f0[i - 1]
                    pitch_after = f0[i + transition_frames - 1] if i + transition_frames - 1 < T else f0[i]

                    # Linear interpolation across transition
                    for j in range(transition_frames):
                        if i + j < T:
                            alpha = j / transition_frames
                            f0_porta[i + j] = pitch_before * (1 - alpha) + pitch_after * alpha
    else:
        # Without phonemes, detect pitch jumps in voiced regions
        voiced_mask = f0 > 0

        for i in range(1, T - transition_frames):
            if voiced_mask[i - 1] and voiced_mask[i]:
                # Check for significant pitch jump (> 5%)
                pitch_ratio = f0[i] / f0[i - 1] if f0[i - 1] > 0 else 1.0
                if abs(pitch_ratio - 1.0) > 0.05:
                    pitch_before = f0[i - 1]
                    pitch_after = f0[i]

                    # Linear interpolation
                    for j in range(transition_frames):
                        if i + j < T:
                            alpha = j / transition_frames
                            f0_porta[i + j] = pitch_before * (1 - alpha) + pitch_after * alpha

    return f0_porta


def add_expression(
    f0: np.ndarray,
    phonemes: Optional[np.ndarray] = None,
    frame_period_ms: float = 16.0,
    # Vibrato params
    enable_vibrato: bool = True,
    vibrato_rate_hz: float = 5.5,
    vibrato_depth: float = 0.03,
    min_note_frames: int = 18,
    onset_delay_frames: int = 9,
    ramp_in_frames: int = 6,
    # Portamento params
    enable_portamento: bool = True,
    transition_frames: int = 3,
) -> np.ndarray:
    """
    Add expressive F0 characteristics (vibrato + portamento).

    Args:
        f0: (T,) F0 in Hz
        phonemes: (T,) optional phoneme IDs for better boundary detection
        frame_period_ms: Frame period in milliseconds
        enable_vibrato: Whether to add vibrato
        vibrato_rate_hz: Vibrato frequency (5-6 Hz typical)
        vibrato_depth: Vibrato depth as fraction (0.02-0.04 typical)
        min_note_frames: Minimum frames for vibrato application
        onset_delay_frames: Frames before vibrato starts
        ramp_in_frames: Frames for vibrato to ramp in
        enable_portamento: Whether to add portamento
        transition_frames: Frames for pitch glide

    Returns:
        f0_expressive: (T,) F0 with expression applied
    """
    f0_out = f0.copy()

    # Apply portamento first (affects boundary transitions)
    if enable_portamento:
        f0_out = add_portamento(
            f0_out,
            phonemes=phonemes,
            transition_frames=transition_frames,
        )

    # Then apply vibrato (on sustained portions)
    if enable_vibrato:
        f0_out = add_vibrato(
            f0_out,
            frame_period_ms=frame_period_ms,
            vibrato_rate_hz=vibrato_rate_hz,
            vibrato_depth=vibrato_depth,
            min_note_frames=min_note_frames,
            onset_delay_frames=onset_delay_frames,
            ramp_in_frames=ramp_in_frames,
        )

    return f0_out


def add_expression_batch(
    f0_batch: np.ndarray,
    phonemes_batch: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply expression to a batch of F0 contours.

    Args:
        f0_batch: (B, T) batch of F0 contours
        phonemes_batch: (B, T) optional batch of phoneme IDs
        **kwargs: arguments passed to add_expression()

    Returns:
        f0_expressive: (B, T) batch with expression applied
    """
    B, T = f0_batch.shape
    f0_out = np.zeros_like(f0_batch)

    for b in range(B):
        phonemes = phonemes_batch[b] if phonemes_batch is not None else None
        f0_out[b] = add_expression(f0_batch[b], phonemes=phonemes, **kwargs)

    return f0_out
