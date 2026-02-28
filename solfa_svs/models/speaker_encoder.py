"""
Speaker Encoder for SolfaSVS.

Wraps a frozen pretrained speaker verification model to extract speaker
embeddings from reference audio. Used for zero-shot voice cloning in the
speaker-conditioned SolfaDiT.

Supports multiple backends (tried in order):
  1. FunASR AutoModel (ModelScope speaker models)
  2. ModelScope Pipeline (speaker-verification task)
  3. wespeaker package

Usage:
    encoder = SpeakerEncoder(device="cuda")
    embedding = encoder.encode_from_file("reference.wav")  # (1, 256)

    # Or from raw waveform
    embedding = encoder(waveform, sr=16000)  # (B, 256)
"""

import os
import warnings
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Optional, Union


class SpeakerEncoder(nn.Module):
    """
    Frozen pretrained speaker encoder + learnable projection.

    Architecture:
        Audio waveform (16kHz mono)
        -> Frozen speaker verification model (192-dim)
        -> Learnable Linear(192, output_dim)
        -> Speaker embedding (B, output_dim)

    The frozen encoder provides generalization to unseen speakers.
    Only the projection layer is trained.
    """

    # Default model: CAM++ from ModelScope, well-supported across backends
    DEFAULT_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"
    EXPECTED_SR = 16000

    def __init__(
        self,
        output_dim: int = 256,
        encoder_dim: int = 192,
        model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            output_dim: Output embedding dimension (default 256)
            encoder_dim: Pretrained encoder output dimension (192 for CAM++)
            model_name: ModelScope model name for speaker encoder
            device: Device for computation
        """
        super().__init__()

        self.output_dim = output_dim
        self.encoder_dim = encoder_dim
        self.device = device
        self._model_name = model_name or self.DEFAULT_MODEL

        # Learnable projection: encoder_dim -> output_dim
        self.projection = nn.Linear(encoder_dim, output_dim)

        # Frozen encoder (loaded lazily to avoid issues during checkpoint loading)
        self._encoder = None
        self._encoder_type = None
        self._encoder_loaded = False

    def _load_encoder(self):
        """Lazily load the pretrained speaker encoder with fallback chain."""
        if self._encoder_loaded:
            return

        # --- Try 1: FunASR AutoModel ---
        try:
            from funasr import AutoModel

            # Always load on CPU to avoid GPU OOM during inference
            # (the full pipeline already occupies most GPU memory).
            # CAM++ is fast enough on CPU (<1s for 30s audio).
            self._encoder = AutoModel(
                model=self._model_name,
                disable_update=True,
                device="cpu",
            )
            self._encoder_type = "funasr"
            self._encoder_loaded = True
            print(f"Speaker encoder loaded via FunASR: {self._model_name}")
            return
        except ImportError:
            pass
        except (AssertionError, Exception) as e:
            warnings.warn(
                f"FunASR failed to load '{self._model_name}': {e}. "
                f"Trying modelscope pipeline..."
            )

        # --- Try 2: ModelScope Pipeline ---
        try:
            from modelscope.pipelines import pipeline as ms_pipeline
            from modelscope.utils.constant import Tasks

            self._encoder = ms_pipeline(
                task=Tasks.speaker_verification,
                model=self._model_name,
            )
            self._encoder_type = "modelscope"
            self._encoder_loaded = True
            print(f"Speaker encoder loaded via ModelScope pipeline: {self._model_name}")
            return
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(
                f"ModelScope pipeline failed for '{self._model_name}': {e}. "
                f"Trying wespeaker..."
            )

        # --- Try 3: wespeaker ---
        try:
            import wespeaker

            self._encoder = wespeaker.load_model_local(self._model_name)
            self._encoder_type = "wespeaker"
            self._encoder_loaded = True
            print(f"Speaker encoder loaded via wespeaker: {self._model_name}")
            return
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"wespeaker failed: {e}")

        raise ImportError(
            f"Failed to load speaker encoder '{self._model_name}' with any backend. "
            f"Tried: funasr, modelscope, wespeaker.\n"
            f"Install with: pip install funasr modelscope\n"
            f"If the model is not registered in your funasr version, try:\n"
            f"  --speaker_encoder_name iic/speech_campplus_sv_zh-cn_16k-common"
        )

    def _extract_raw_embedding(
        self,
        waveform: torch.Tensor,
        sr: int = 16000,
    ) -> torch.Tensor:
        """
        Extract raw speaker embedding from the frozen encoder.

        Args:
            waveform: (B, T) or (T,) audio at 16kHz mono
            sr: Sample rate (will resample if not 16kHz)

        Returns:
            (B, encoder_dim) raw speaker embedding
        """
        self._load_encoder()

        # Ensure correct sample rate
        if sr != self.EXPECTED_SR:
            resampler = torchaudio.transforms.Resample(sr, self.EXPECTED_SR)
            waveform = resampler(waveform)

        # Ensure batch dimension
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]

        if self._encoder_type == "funasr":
            raw_emb = self._extract_funasr(waveform, batch_size)
        elif self._encoder_type == "modelscope":
            raw_emb = self._extract_modelscope(waveform, batch_size)
        elif self._encoder_type == "wespeaker":
            raw_emb = self._extract_wespeaker(waveform, batch_size)
        else:
            raise RuntimeError(f"Unknown encoder type: {self._encoder_type}")

        # Ensure correct shape
        if raw_emb.ndim == 1:
            raw_emb = raw_emb.unsqueeze(0)

        # Truncate or pad to encoder_dim if model output differs
        if raw_emb.shape[-1] != self.encoder_dim:
            if raw_emb.shape[-1] > self.encoder_dim:
                raw_emb = raw_emb[:, :self.encoder_dim]
            else:
                pad = torch.zeros(
                    raw_emb.shape[0],
                    self.encoder_dim - raw_emb.shape[-1],
                    device=raw_emb.device,
                )
                raw_emb = torch.cat([raw_emb, pad], dim=-1)

        return raw_emb.to(self.projection.weight.device)

    def _extract_funasr(self, waveform: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Extract embeddings via FunASR."""
        embeddings = []
        for i in range(batch_size):
            wav_np = waveform[i].cpu().numpy()
            result = self._encoder.generate(input=wav_np, output_dir=None)
            if isinstance(result, list) and len(result) > 0:
                emb = result[0].get("spk_embedding", result[0].get("embedding"))
            else:
                emb = result
            embeddings.append(self._to_tensor(emb))
        return torch.stack(embeddings, dim=0)

    def _extract_modelscope(self, waveform: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Extract embeddings via ModelScope speaker-verification pipeline."""
        import tempfile
        import soundfile as sf

        embeddings = []
        for i in range(batch_size):
            wav_np = waveform[i].cpu().numpy()

            # ModelScope pipeline expects file paths; write to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, wav_np, self.EXPECTED_SR)

            try:
                # The pipeline expects two audio files for verification.
                # We pass the same file twice and extract the embedding from
                # the pipeline's internal model.
                result = self._encoder([tmp_path, tmp_path])

                # Try to get embedding from result dict
                emb = None
                if isinstance(result, dict):
                    for key in ("embs", "spk_embedding", "embedding"):
                        if key in result:
                            val = result[key]
                            if isinstance(val, list) and len(val) > 0:
                                emb = val[0]
                            else:
                                emb = val
                            break

                # If pipeline didn't return embeddings, extract from model
                if emb is None:
                    emb = self._extract_from_pipeline_model(tmp_path)

                embeddings.append(self._to_tensor(emb))
            finally:
                os.unlink(tmp_path)

        return torch.stack(embeddings, dim=0)

    def _extract_from_pipeline_model(self, audio_path: str) -> torch.Tensor:
        """
        Extract embedding by accessing the pipeline's internal model.

        This handles the case where the pipeline's __call__ only returns
        a similarity score without exposing the embeddings.
        """
        pipeline = self._encoder

        # Access the underlying model and preprocessor
        model = getattr(pipeline, "model", None)
        preprocessor = getattr(pipeline, "preprocessor", None)

        if model is None:
            raise RuntimeError(
                "Cannot extract embeddings: ModelScope pipeline has no .model attribute. "
                "Try using a different speaker encoder model."
            )

        # Try calling the model directly with the audio path
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")

        # Try common forward signatures
        try:
            # Some models accept raw audio tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            if hasattr(model, "extract_embedding"):
                emb = model.extract_embedding(audio_tensor)
            elif hasattr(model, "forward"):
                emb = model.forward(audio_tensor)
            else:
                emb = model(audio_tensor)
            return self._to_tensor(emb)
        except Exception:
            pass

        # Try preprocessor -> model path
        if preprocessor is not None:
            try:
                inputs = preprocessor(audio_path)
                emb = model(inputs)
                return self._to_tensor(emb)
            except Exception:
                pass

        raise RuntimeError(
            "Failed to extract embedding from ModelScope pipeline model. "
            "Try: --speaker_encoder_name iic/speech_campplus_sv_zh-cn_16k-common"
        )

    def _extract_wespeaker(self, waveform: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Extract embeddings via wespeaker."""
        embeddings = []
        for i in range(batch_size):
            wav_np = waveform[i].cpu().numpy()
            emb = self._encoder.extract_embedding_wav(wav_np, sample_rate=self.EXPECTED_SR)
            embeddings.append(self._to_tensor(emb))
        return torch.stack(embeddings, dim=0)

    @staticmethod
    def _to_tensor(emb) -> torch.Tensor:
        """Convert any embedding format to a 1-D float tensor."""
        if isinstance(emb, torch.Tensor):
            return emb.squeeze().float()
        if isinstance(emb, dict):
            # Some models return dicts — try common keys
            for key in ("spk_embedding", "embedding", "emb"):
                if key in emb:
                    return SpeakerEncoder._to_tensor(emb[key])
            raise ValueError(f"Cannot find embedding in dict with keys: {list(emb.keys())}")
        return torch.from_numpy(np.array(emb)).float().squeeze()

    def forward(
        self,
        waveform: torch.Tensor,
        sr: int = 16000,
    ) -> torch.Tensor:
        """
        Extract projected speaker embedding.

        Args:
            waveform: (B, T) or (T,) audio waveform
            sr: Sample rate

        Returns:
            (B, output_dim) speaker embedding
        """
        with torch.no_grad():
            raw_emb = self._extract_raw_embedding(waveform, sr)

        # Project through learnable layer
        return self.projection(raw_emb)

    @torch.no_grad()
    def encode_from_file(
        self,
        audio_path: str,
        max_duration_sec: float = 30.0,
    ) -> torch.Tensor:
        """
        Extract speaker embedding from an audio file.

        Args:
            audio_path: Path to audio file
            max_duration_sec: Maximum audio duration to use (truncates longer files)

        Returns:
            (1, output_dim) speaker embedding tensor
        """
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        # Truncate if too long
        max_samples = int(max_duration_sec * sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        return self.forward(waveform.unsqueeze(0), sr=sr)

    @torch.no_grad()
    def encode_batch_from_files(
        self,
        audio_paths: list,
        max_duration_sec: float = 30.0,
    ) -> torch.Tensor:
        """
        Extract speaker embeddings from multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            max_duration_sec: Maximum audio duration per file

        Returns:
            (B, output_dim) speaker embeddings
        """
        embeddings = []
        for path in audio_paths:
            emb = self.encode_from_file(path, max_duration_sec)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

    @staticmethod
    def get_zero_embedding(batch_size: int, output_dim: int = 256) -> torch.Tensor:
        """Get zero speaker embedding for unconditional generation."""
        return torch.zeros(batch_size, output_dim)
