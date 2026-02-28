"""
WORLD Vocoder Feature Extraction

Extracts high-quality speech features using WORLD vocoder:
- F0 (fundamental frequency)
- SP (spectral envelope) - captures vocal tract shape
- AP (aperiodicity) - captures breathiness/noise

These features can be used for analysis-synthesis and voice conversion.
"""

import numpy as np
import pyworld as pw
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


class WORLDProcessor:
    """
    WORLD vocoder feature extractor and synthesizer.

    WORLD is a high-quality vocoder that separates speech into:
    - F0: Pitch contour
    - SP: Spectral envelope (vocal tract filter)
    - AP: Aperiodicity (breathiness/noise component)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_period: float = 16.0,  # ms (matches hop_length=256 at 16kHz)
        f0_floor: float = 71.0,  # ~D2 (MIDI 38)
        f0_ceil: float = 1100.0,  # >C6 (MIDI 84, ~1047Hz) - aligned with VPR converter vocal limit
        fft_size: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil

        # FFT size for spectral envelope
        # WORLD recommends 2^n where n >= log2(3*sample_rate/f0_floor)
        if fft_size is None:
            self.fft_size = pw.get_cheaptrick_fft_size(sample_rate, f0_floor)
        else:
            self.fft_size = fft_size

        # Spectral envelope dimensions (fft_size // 2 + 1)
        self.sp_dim = self.fft_size // 2 + 1

    def extract_features(
        self,
        audio: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract WORLD features from audio.

        Args:
            audio: (T,) audio waveform (float64, range [-1, 1])

        Returns:
            f0: (T_frames,) F0 in Hz (0 for unvoiced)
            sp: (T_frames, sp_dim) Spectral envelope (log-scale)
            ap: (T_frames, sp_dim) Aperiodicity
        """
        # Ensure correct dtype
        if audio.dtype != np.float64:
            audio = audio.astype(np.float64)

        # Extract F0 using DIO + StoneMask (high quality)
        f0, timeaxis = pw.dio(
            audio,
            self.sample_rate,
            f0_floor=self.f0_floor,
            f0_ceil=self.f0_ceil,
            frame_period=self.frame_period,
        )

        # Refine F0 using StoneMask
        f0 = pw.stonemask(audio, f0, timeaxis, self.sample_rate)

        # Extract spectral envelope using CheapTrick
        sp = pw.cheaptrick(audio, f0, timeaxis, self.sample_rate, fft_size=self.fft_size)

        # Extract aperiodicity using D4C
        ap = pw.d4c(audio, f0, timeaxis, self.sample_rate, fft_size=self.fft_size)

        return f0, sp, ap

    def synthesize(
        self,
        f0: np.ndarray,
        sp: np.ndarray,
        ap: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize audio from WORLD features.

        Args:
            f0: (T_frames,) F0 in Hz
            sp: (T_frames, sp_dim) Spectral envelope
            ap: (T_frames, sp_dim) Aperiodicity

        Returns:
            audio: (T,) synthesized audio waveform (float64)
        """
        # Ensure correct dtypes and C-contiguous arrays (required by pyworld)
        f0 = np.ascontiguousarray(f0, dtype=np.float64)
        sp = np.ascontiguousarray(sp, dtype=np.float64)
        ap = np.ascontiguousarray(ap, dtype=np.float64)

        # Synthesize using WORLD
        audio = pw.synthesize(f0, sp, ap, self.sample_rate, self.frame_period)

        return audio

    def sp_to_log(self, sp: np.ndarray) -> np.ndarray:
        """Convert spectral envelope to log scale."""
        return np.log(np.clip(sp, a_min=1e-10, a_max=None))

    def log_to_sp(self, log_sp: np.ndarray) -> np.ndarray:
        """Convert log spectral envelope to linear scale."""
        return np.exp(log_sp)

    def extract_energy(self, sp: np.ndarray) -> np.ndarray:
        """
        Extract frame-level energy from spectral envelope.

        Args:
            sp: (T_frames, sp_dim) Spectral envelope

        Returns:
            energy: (T_frames,) Frame-level energy (log scale)
        """
        # Sum energy across frequency bins
        energy = np.sum(sp, axis=1)
        # Log scale
        energy = np.log(np.clip(energy, a_min=1e-10, a_max=None))
        return energy


# =============================================================================
# MIDI Timing Utilities
# =============================================================================

PPQN = 480  # Ticks per quarter note (standard MIDI)


def tick_to_ms(
    tick: int,
    tempo_map: list,
    measure0_tick: int = 0
) -> float:
    """
    Convert MIDI tick to milliseconds using tempo map.

    Args:
        tick: Absolute tick position
        tempo_map: List of (tick, tempo_us) tuples sorted by tick
        measure0_tick: Tick position of measure 0 (audio start reference)

    Returns:
        Time in milliseconds relative to audio start
    """
    # Adjust for measure 0 offset
    relative_tick = tick - measure0_tick
    if relative_tick < 0:
        return 0.0

    ms = 0.0
    prev_tick = 0
    prev_tempo = tempo_map[0][1] if tempo_map else 500000  # Default 120 BPM

    for tempo_tick, tempo_us in tempo_map:
        # Adjust tempo tick relative to measure 0
        adj_tempo_tick = tempo_tick - measure0_tick
        if adj_tempo_tick < 0:
            prev_tempo = tempo_us
            continue

        if adj_tempo_tick >= relative_tick:
            break

        # Calculate ms for segment at previous tempo
        segment_ticks = adj_tempo_tick - prev_tick
        if segment_ticks > 0:
            ms += (segment_ticks * prev_tempo / 1000) / PPQN

        prev_tick = adj_tempo_tick
        prev_tempo = tempo_us

    # Calculate remaining ticks at current tempo
    remaining = relative_tick - prev_tick
    if remaining > 0:
        ms += (remaining * prev_tempo / 1000) / PPQN

    return ms


def build_tempo_map(midi_data: dict) -> list:
    """
    Build tempo map from midi.json tempos array.

    Args:
        midi_data: Loaded midi.json data

    Returns:
        List of (tick, tempo_us) tuples sorted by tick
    """
    tempos = midi_data.get('tempos', [])
    if not tempos:
        return [(0, 500000)]  # Default 120 BPM

    tempo_map = [(t['tick'], t['tempo']) for t in tempos]
    tempo_map.sort(key=lambda x: x[0])
    return tempo_map


def find_measure0_tick(midi_data: dict) -> int:
    """
    Find the tick position of Measure 0 (audio start reference).

    Args:
        midi_data: Loaded midi.json data

    Returns:
        Tick position of measure 0
    """
    measures = midi_data.get('measures', {})
    for tick_str, measure_info in measures.items():
        if measure_info.get('measure') == 0:
            return int(tick_str)
    return 0


def test_world_processor():
    """Test WORLD feature extraction and synthesis."""
    import matplotlib.pyplot as plt

    # Load test audio
    audio, sr = sf.read('data/processed/humming/audio/1146640.wav')

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        import scipy.signal as signal
        audio = signal.resample_poly(audio, 16000, sr)
        sr = 16000

    # Create processor
    processor = WORLDProcessor(sample_rate=sr)

    print(f"Sample rate: {sr} Hz")
    print(f"Audio length: {len(audio)} samples ({len(audio)/sr:.2f}s)")
    print(f"FFT size: {processor.fft_size}")
    print(f"SP dimension: {processor.sp_dim}")

    # Extract features
    f0, sp, ap = processor.extract_features(audio)

    print(f"\nExtracted features:")
    print(f"  F0: {f0.shape}, range: [{f0.min():.1f}, {f0.max():.1f}] Hz")
    print(f"  SP: {sp.shape}")
    print(f"  AP: {ap.shape}")

    # Convert to log scale
    log_sp = processor.sp_to_log(sp)
    print(f"  Log SP range: [{log_sp.min():.2f}, {log_sp.max():.2f}]")

    # Extract energy
    energy = processor.extract_energy(sp)
    print(f"  Energy: {energy.shape}, range: [{energy.min():.2f}, {energy.max():.2f}]")

    # Synthesize
    audio_synth = processor.synthesize(f0, sp, ap)

    print(f"\nSynthesized audio: {len(audio_synth)} samples")

    # Save for comparison
    sf.write('/tmp/world_original.wav', audio, sr)
    sf.write('/tmp/world_resynthesis.wav', audio_synth, sr)

    print(f"\nSaved:")
    print(f"  /tmp/world_original.wav")
    print(f"  /tmp/world_resynthesis.wav")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    time = np.arange(len(f0)) * processor.frame_period / 1000

    # F0
    axes[0].plot(time, f0, 'b-', linewidth=0.8)
    axes[0].set_title('F0 Contour')
    axes[0].set_ylabel('F0 (Hz)')
    axes[0].set_ylim(0, 800)
    axes[0].grid(True, alpha=0.3)

    # Spectral envelope (log)
    im = axes[1].imshow(
        log_sp.T,
        aspect='auto',
        origin='lower',
        extent=[0, time[-1], 0, processor.sp_dim],
        cmap='viridis'
    )
    axes[1].set_title('Spectral Envelope (log scale)')
    axes[1].set_ylabel('Frequency bin')
    plt.colorbar(im, ax=axes[1])

    # Aperiodicity
    im2 = axes[2].imshow(
        ap.T,
        aspect='auto',
        origin='lower',
        extent=[0, time[-1], 0, processor.sp_dim],
        cmap='hot'
    )
    axes[2].set_title('Aperiodicity')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency bin')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('/tmp/world_features.png', dpi=150)
    print(f"  /tmp/world_features.png")


if __name__ == '__main__':
    test_world_processor()
