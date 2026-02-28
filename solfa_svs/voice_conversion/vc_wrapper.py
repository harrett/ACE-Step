"""
Voice Conversion Wrapper for SolfaSVS.

Thin wrapper around RVC (Retrieval-based Voice Conversion) for post-generation
timbre migration. Accepts SolfaSVS output WAV + target speaker model, returns
timbre-converted WAV.

Usage (Phase 1 prototype):
    MIDI --> SolfaSVS (unchanged) --> single-speaker WAV --> RVC --> target-timbre WAV

Supports two backends:
    - RVC v2: Fast, supports f0-preserving mode
    - SO-VITS-SVC 4.1: Higher quality, slower

Prerequisites:
    pip install rvc-python  # For RVC backend
    # or
    pip install so-vits-svc-fork  # For SO-VITS-SVC backend
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


def convert_rvc(
    input_audio: Union[str, np.ndarray],
    model_path: str,
    input_sr: int = 44100,
    output_sr: int = 44100,
    f0_method: str = "rmvpe",
    f0_up_key: int = 0,
    index_path: Optional[str] = None,
    index_rate: float = 0.75,
    protect: float = 0.33,
    device: str = "cpu",
) -> Tuple[int, np.ndarray]:
    """
    Apply RVC voice conversion to audio.

    Args:
        input_audio: Path to WAV file or numpy array (mono, float32)
        model_path: Path to RVC .pth model file
        input_sr: Sample rate of input (if numpy array)
        output_sr: Desired output sample rate
        f0_method: Pitch extraction method ('rmvpe', 'crepe', 'harvest')
        f0_up_key: Semitone shift (0 = preserve original pitch)
        index_path: Optional path to .index file for retrieval
        index_rate: Blend ratio for index retrieval (0-1)
        protect: Protect voiceless consonants (0-0.5, higher = more protection)
        device: Computation device

    Returns:
        (sample_rate, waveform) tuple where waveform is numpy float32
    """
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        raise ImportError(
            "RVC backend requires rvc-python. Install with: pip install rvc-python"
        )

    rvc = RVCInference(device=device)
    rvc.load_model(model_path)

    rvc.set_params(
        f0method=f0_method,
        f0up_key=f0_up_key,
        index_rate=index_rate,
        protect=protect,
    )

    if index_path and os.path.exists(index_path):
        rvc.set_params(index_path=index_path)

    # Handle numpy array input
    tmp_input = None
    if isinstance(input_audio, np.ndarray):
        import soundfile as sf

        tmp_input = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_input.name, input_audio, input_sr)
        input_path = tmp_input.name
    else:
        input_path = input_audio

    tmp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_output.close()

    try:
        rvc.infer_file(input_path, tmp_output.name)

        import soundfile as sf
        audio_out, sr_out = sf.read(tmp_output.name, dtype="float32")

        # Resample if needed
        if sr_out != output_sr:
            import torchaudio
            import torch

            audio_tensor = torch.from_numpy(audio_out).float()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.T  # (channels, samples)
            resampler = torchaudio.transforms.Resample(sr_out, output_sr)
            audio_tensor = resampler(audio_tensor)
            audio_out = audio_tensor.squeeze().numpy()
            sr_out = output_sr

        return sr_out, audio_out

    finally:
        if tmp_input:
            os.unlink(tmp_input.name)
        os.unlink(tmp_output.name)


def convert_sovits(
    input_audio: Union[str, np.ndarray],
    model_path: str,
    input_sr: int = 44100,
    output_sr: int = 44100,
    speaker: str = "default",
    transpose: int = 0,
    auto_predict_f0: bool = False,
    device: str = "cpu",
) -> Tuple[int, np.ndarray]:
    """
    Apply SO-VITS-SVC voice conversion to audio.

    Args:
        input_audio: Path to WAV file or numpy array (mono, float32)
        model_path: Path to SO-VITS-SVC model directory
        input_sr: Sample rate of input (if numpy array)
        output_sr: Desired output sample rate
        speaker: Speaker name in the model
        transpose: Semitone shift (0 = preserve)
        auto_predict_f0: Let model predict F0 instead of using input
        device: Computation device

    Returns:
        (sample_rate, waveform) tuple
    """
    try:
        from so_vits_svc_fork.inference.main import infer as sovits_infer
    except ImportError:
        raise ImportError(
            "SO-VITS-SVC backend requires so-vits-svc-fork. "
            "Install with: pip install so-vits-svc-fork"
        )

    # Handle numpy array input
    tmp_input = None
    if isinstance(input_audio, np.ndarray):
        import soundfile as sf

        tmp_input = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_input.name, input_audio, input_sr)
        input_path = tmp_input.name
    else:
        input_path = input_audio

    tmp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_output.close()

    try:
        sovits_infer(
            input_path=Path(input_path),
            output_path=Path(tmp_output.name),
            model_path=Path(model_path),
            speaker=speaker,
            transpose=transpose,
            auto_predict_f0=auto_predict_f0,
            device=device,
        )

        import soundfile as sf
        audio_out, sr_out = sf.read(tmp_output.name, dtype="float32")

        if sr_out != output_sr:
            import torchaudio
            import torch

            audio_tensor = torch.from_numpy(audio_out).float()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.T
            resampler = torchaudio.transforms.Resample(sr_out, output_sr)
            audio_tensor = resampler(audio_tensor)
            audio_out = audio_tensor.squeeze().numpy()
            sr_out = output_sr

        return sr_out, audio_out

    finally:
        if tmp_input:
            os.unlink(tmp_input.name)
        os.unlink(tmp_output.name)


class VoiceConverter:
    """
    Unified voice conversion interface for SolfaSVS.

    Wraps RVC or SO-VITS-SVC backend for post-generation timbre migration.

    Example:
        vc = VoiceConverter(model_path="models/singer_a.pth", backend="rvc")
        sr, wav = vc.convert("outputs/solfa_output.wav")
    """

    BACKENDS = ("rvc", "sovits")

    def __init__(
        self,
        model_path: str,
        backend: str = "rvc",
        index_path: Optional[str] = None,
        device: str = "cpu",
        f0_method: str = "rmvpe",
        protect: float = 0.33,
    ):
        """
        Args:
            model_path: Path to VC model (.pth for RVC, directory for SO-VITS-SVC)
            backend: "rvc" or "sovits"
            index_path: Optional .index file for RVC retrieval
            device: Computation device
            f0_method: Pitch extraction method for RVC
            protect: Voiceless consonant protection for RVC
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from {self.BACKENDS}")

        self.model_path = model_path
        self.backend = backend
        self.index_path = index_path
        self.device = device
        self.f0_method = f0_method
        self.protect = protect

    def convert(
        self,
        input_audio: Union[str, np.ndarray],
        input_sr: int = 44100,
        output_sr: int = 44100,
        f0_up_key: int = 0,
    ) -> Tuple[int, np.ndarray]:
        """
        Convert input audio to target speaker timbre.

        Args:
            input_audio: Path to WAV or numpy array (mono float32)
            input_sr: Sample rate of input (when numpy)
            output_sr: Desired output sample rate
            f0_up_key: Semitone shift (0 = preserve pitch)

        Returns:
            (sample_rate, waveform) tuple
        """
        if self.backend == "rvc":
            return convert_rvc(
                input_audio=input_audio,
                model_path=self.model_path,
                input_sr=input_sr,
                output_sr=output_sr,
                f0_method=self.f0_method,
                f0_up_key=f0_up_key,
                index_path=self.index_path,
                protect=self.protect,
                device=self.device,
            )
        else:
            return convert_sovits(
                input_audio=input_audio,
                model_path=self.model_path,
                input_sr=input_sr,
                output_sr=output_sr,
                transpose=f0_up_key,
                device=self.device,
            )
