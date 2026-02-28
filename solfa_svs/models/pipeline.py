"""
SolfaSVS Inference Pipeline.

End-to-end inference: MIDI → DCAE latent diffusion → Audio output.

Input: .mid files with sol-fa lyric events from midi_for_ace/ format.
Only sheet music is needed — no audio information.

Uses:
- Trained SolfaDiT + MidiEncoder for latent generation
- Frozen ACE-Step DCAE decoder + vocoder for audio synthesis
- Flow-matching Euler scheduler with classifier-free guidance
- APG (Asymmetric Prompt Guidance) for improved generation quality
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from typing import Optional, List, Dict

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor

from solfa_svs.models.solfa_dit import SolfaDiT
from solfa_svs.models.midi_encoder import MidiEncoder
from solfa_svs.models.speaker_encoder import SpeakerEncoder
from solfa_svs.data.midi_parser import (
    load_midi_file,
    synthesize_features_from_notes,
)
from solfa_svs.data.f0_expression import add_expression
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.apg_guidance import apg_forward, MomentumBuffer


# DCAE latent frame rate
LATENT_FRAME_RATE = 44100 / 512 / 8  # ≈ 10.77 fps


class SolfaSVSPipeline:
    """
    End-to-end inference pipeline for SolfaSVS.

    Components:
    - solfa_dit: Trained diffusion transformer
    - midi_encoder: Trained MIDI conditioning encoder
    - dcae: Frozen ACE-Step DCAE decoder + vocoder
    - speaker_encoder: Optional frozen speaker encoder for zero-shot voice cloning
    """

    def __init__(
        self,
        solfa_dit: SolfaDiT,
        midi_encoder: MidiEncoder,
        dcae: MusicDCAE,
        speaker_encoder: Optional[SpeakerEncoder] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.solfa_dit = solfa_dit.to(device).eval()
        self.midi_encoder = midi_encoder.to(device).eval()
        self.dcae = dcae.to(device).eval()
        self.speaker_encoder = speaker_encoder
        if self.speaker_encoder is not None:
            self.speaker_encoder = self.speaker_encoder.to(device).eval()
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        dcae_checkpoint_dir: Optional[str] = None,
        speaker_projection_path: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Load pipeline from a training checkpoint.

        Args:
            checkpoint_path: Path to PyTorch Lightning checkpoint
            dcae_checkpoint_dir: Path to ACE-Step checkpoint for DCAE
            speaker_projection_path: Path to speaker_projection.pt saved during
                preprocessing. Required for speaker-conditioned models to produce
                correct embeddings at inference time.
            device: Device for inference
            dtype: Dtype for inference
        """
        from solfa_svs.training.trainer import SolfaSVSTrainer

        # Load training module
        trainer_module = SolfaSVSTrainer.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        solfa_dit = trainer_module.solfa_dit.to(dtype)
        midi_encoder = trainer_module.midi_encoder.to(dtype)

        # Check if model has speaker conditioning
        speaker_encoder = None
        speaker_dim = trainer_module.hparams.get("speaker_dim", 0)
        if speaker_dim > 0:
            # Create speaker encoder with matching output_dim
            speaker_encoder = SpeakerEncoder(
                output_dim=speaker_dim,
                device=device,
            )
            # Load the projection weights that were used during preprocessing.
            # Without these, the 192→speaker_dim projection is random and the
            # speaker embeddings will be in a different space than training.
            if speaker_projection_path and os.path.exists(speaker_projection_path):
                proj_state = torch.load(
                    speaker_projection_path, map_location="cpu", weights_only=True
                )
                speaker_encoder.projection.load_state_dict(proj_state)
                print(f"Loaded speaker projection from: {speaker_projection_path}")
            else:
                import warnings
                warnings.warn(
                    "Speaker-conditioned model loaded WITHOUT speaker_projection.pt. "
                    "Speaker embeddings will use random projection weights and will NOT "
                    "match training. Pass --speaker_projection_path to fix this."
                )

        # Load DCAE
        if dcae_checkpoint_dir is not None:
            dcae_checkpoint_dir = os.path.abspath(dcae_checkpoint_dir)
            dcae = MusicDCAE(
                dcae_checkpoint_path=os.path.join(dcae_checkpoint_dir, "music_dcae_f8c8"),
                vocoder_checkpoint_path=os.path.join(dcae_checkpoint_dir, "music_vocoder"),
            )
        else:
            dcae = MusicDCAE()
        dcae = dcae.to(torch.float32)
        dcae.requires_grad_(False)

        return cls(solfa_dit, midi_encoder, dcae, speaker_encoder, device, dtype)

    @torch.no_grad()
    def generate_from_midi(
        self,
        midi_path: str,
        metadata_path: Optional[str] = None,
        num_inference_steps: int = 60,
        guidance_scale: float = 5.0,
        omega_scale: float = 10.0,
        output_sr: int = 44100,
        seed: Optional[int] = None,
        apply_expression: bool = True,
        reference_audio: Optional[str] = None,
    ) -> tuple:
        """
        Generate audio from a MIDI file (production path).

        Only the sheet music is needed — no audio features.
        Accepts .mid files from midi_for_ace/ format with embedded sol-fa
        lyric events (Chinese pinyin: Dao, Rei, Mi, Fa, So, La, Xi).

        Args:
            midi_path: Path to .mid MIDI file with sol-fa lyrics
            metadata_path: Optional path to companion .json metadata file
            num_inference_steps: Denoising steps (default 60)
            guidance_scale: CFG scale (default 5.0)
            omega_scale: Mean-shift scale for scheduler
            output_sr: Output sample rate (default 44100)
            seed: Random seed
            apply_expression: Apply vibrato/portamento to F0
            reference_audio: Optional path to reference audio for speaker cloning

        Returns:
            (sample_rate, waveform) tuple
        """
        # Load MIDI with sol-fa lyric events
        features = load_midi_file(
            midi_path,
            metadata_path=metadata_path,
            frame_rate=LATENT_FRAME_RATE,
        )

        f0 = features["f0"]
        energy = features["energy"]
        phonemes = features["phonemes"]
        notes = features["notes"]

        # Apply expression (vibrato/portamento)
        if apply_expression and len(f0) > 0:
            frame_period_ms = 1000.0 / LATENT_FRAME_RATE
            f0 = add_expression(
                f0, phonemes=phonemes,
                frame_period_ms=frame_period_ms,
                min_note_frames=3,
                onset_delay_frames=2,
                ramp_in_frames=2,
                transition_frames=1,
            )

        return self._run_diffusion(
            f0=f0, energy=energy, phonemes=phonemes, notes=notes,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            output_sr=output_sr,
            seed=seed,
            reference_audio=reference_audio,
        )

    @torch.no_grad()
    def generate_from_notes(
        self,
        notes: List[Dict],
        duration_sec: float,
        num_inference_steps: int = 60,
        guidance_scale: float = 5.0,
        omega_scale: float = 10.0,
        output_sr: int = 44100,
        seed: Optional[int] = None,
        apply_expression: bool = True,
        reference_audio: Optional[str] = None,
    ) -> tuple:
        """
        Generate audio from a list of note events (programmatic API).

        Args:
            notes: List of note dicts, each with:
                solfa (str): sol-fa syllable ('Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Ti')
                midi_pitch (int): MIDI note number
                onset_sec (float): note onset in seconds
                offset_sec (float): note offset in seconds
                velocity (int, optional): 0-127, default 80
            duration_sec: Total duration in seconds
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            omega_scale: Mean-shift scale
            output_sr: Output sample rate
            seed: Random seed
            apply_expression: Apply vibrato/portamento
            reference_audio: Optional path to reference audio for speaker cloning

        Returns:
            (sample_rate, waveform) tuple
        """
        f0, energy, phonemes, notes_enriched = synthesize_features_from_notes(
            notes, duration_sec, frame_rate=LATENT_FRAME_RATE,
        )

        if apply_expression and len(f0) > 0:
            frame_period_ms = 1000.0 / LATENT_FRAME_RATE
            f0 = add_expression(
                f0, phonemes=phonemes,
                frame_period_ms=frame_period_ms,
                min_note_frames=3,
                onset_delay_frames=2,
                ramp_in_frames=2,
                transition_frames=1,
            )

        return self._run_diffusion(
            f0=f0, energy=energy, phonemes=phonemes, notes=notes_enriched,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            output_sr=output_sr,
            seed=seed,
            reference_audio=reference_audio,
        )

    @torch.no_grad()
    def _run_diffusion(
        self,
        f0: np.ndarray,
        energy: np.ndarray,
        phonemes: np.ndarray,
        notes: List[Dict],
        num_inference_steps: int = 60,
        guidance_scale: float = 5.0,
        omega_scale: float = 10.0,
        output_sr: int = 44100,
        seed: Optional[int] = None,
        reference_audio: Optional[str] = None,
    ) -> tuple:
        """
        Core diffusion loop shared by all generate methods.

        Args:
            f0: (L,) F0 in Hz at latent frame rate
            energy: (L,) energy at latent frame rate
            phonemes: (L,) phoneme IDs at latent frame rate
            notes: List of note event dicts
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            omega_scale: Mean-shift scale for scheduler
            output_sr: Output sample rate
            seed: Random seed
            reference_audio: Optional path to reference audio for speaker cloning

        Returns:
            (sample_rate, waveform) tuple
        """
        device = self.device
        dtype = self.dtype
        L = len(f0)

        # Prepare tensors
        f0_t = torch.from_numpy(f0).to(device=device, dtype=dtype).unsqueeze(0)
        energy_t = torch.from_numpy(energy).to(device=device, dtype=dtype).unsqueeze(0)
        phonemes_t = torch.from_numpy(phonemes).long().unsqueeze(0).to(device)
        attention_mask = torch.ones(1, L, device=device, dtype=dtype)

        # Prepare note tensors
        total_dur = L / LATENT_FRAME_RATE if L > 0 else 1.0
        max_notes = max(len(notes), 1)
        note_phonemes = torch.zeros(1, max_notes, dtype=torch.long, device=device)
        note_pitches = torch.zeros(1, max_notes, dtype=torch.long, device=device)
        note_velocities = torch.zeros(1, max_notes, dtype=torch.long, device=device)
        note_durations = torch.zeros(1, max_notes, device=device, dtype=dtype)
        note_positions = torch.zeros(1, max_notes, device=device, dtype=dtype)
        note_mask = torch.zeros(1, max_notes, device=device, dtype=dtype)

        for j, note in enumerate(notes):
            note_phonemes[0, j] = note["phoneme_id"]
            note_pitches[0, j] = note["midi_pitch"]
            note_velocities[0, j] = note["velocity"]
            note_durations[0, j] = note["duration_sec"]
            note_positions[0, j] = note["onset_sec"] / total_dur
            note_mask[0, j] = 1.0

        # MIDI encoder → conditioning
        encoder_hidden_states, encoder_mask = self.midi_encoder(
            note_phonemes=note_phonemes,
            note_pitches=note_pitches,
            note_velocities=note_velocities,
            note_durations=note_durations,
            note_positions=note_positions,
            note_mask=note_mask,
            f0=f0_t,
            energy=energy_t,
            phonemes=phonemes_t,
            attention_mask=attention_mask,
        )
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        # Extract speaker embedding from reference audio
        speaker_embeds = None
        null_speaker_embeds = None
        has_speaker = (
            self.speaker_encoder is not None
            and self.solfa_dit.speaker_embedding_dim > 0
        )
        if has_speaker and reference_audio is not None:
            # Move projection to CPU for compatibility with FunASR output,
            # then move the result to the target device.
            self.speaker_encoder.projection = self.speaker_encoder.projection.cpu()
            speaker_embeds = self.speaker_encoder.encode_from_file(
                reference_audio
            ).to(device=device, dtype=dtype)
            # Move projection back to device for potential future calls
            self.speaker_encoder.projection = self.speaker_encoder.projection.to(device)
            null_speaker_embeds = torch.zeros_like(speaker_embeds)
            print(f"  Speaker embedding: norm={speaker_embeds.float().norm():.4f}")
        elif has_speaker:
            # No reference audio: use zero embedding (unconditional)
            spk_dim = self.solfa_dit.speaker_embedding_dim
            speaker_embeds = torch.zeros(1, spk_dim, device=device, dtype=dtype)
            null_speaker_embeds = torch.zeros_like(speaker_embeds)

        # Null conditioning for CFG
        null_hidden_states = torch.zeros_like(encoder_hidden_states)
        null_mask = torch.zeros_like(encoder_mask)

        # Initialize noise
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        latent = randn_tensor(
            shape=(1, 8, 16, L),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # Setup scheduler
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=3.0,
        )
        timesteps, _ = retrieve_timesteps(
            scheduler, num_inference_steps=num_inference_steps, device=device,
        )

        do_cfg = guidance_scale > 1.0
        attention_mask_double = (
            torch.cat([attention_mask, attention_mask], dim=0) if do_cfg
            else attention_mask
        )
        momentum_buffer = MomentumBuffer() if do_cfg else None

        # Denoising loop
        for t in timesteps:
            if do_cfg:
                latent_input = torch.cat([latent, latent], dim=0)
                enc_states = torch.cat(
                    [encoder_hidden_states, null_hidden_states], dim=0
                )
                enc_mask = torch.cat([encoder_mask, null_mask], dim=0)
                t_expand = t.expand(2)
                # For CFG: conditional uses real speaker, unconditional uses zeros
                spk_for_dit = None
                if speaker_embeds is not None:
                    spk_for_dit = torch.cat(
                        [speaker_embeds, null_speaker_embeds], dim=0
                    )
            else:
                latent_input = latent
                enc_states = encoder_hidden_states
                enc_mask = encoder_mask
                t_expand = t.expand(1)
                spk_for_dit = speaker_embeds

            noise_pred = self.solfa_dit(
                hidden_states=latent_input,
                attention_mask=attention_mask_double,
                encoder_hidden_states=enc_states,
                encoder_hidden_mask=enc_mask,
                timestep=t_expand.to(dtype),
                speaker_embeds=spk_for_dit,
            )

            if do_cfg:
                pred_cond, pred_uncond = noise_pred.chunk(2)
                noise_pred = apg_forward(
                    pred_cond=pred_cond,
                    pred_uncond=pred_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum_buffer,
                )

            latent = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latent,
                return_dict=False,
                omega=omega_scale,
            )[0]

        # DCAE decode → audio
        _, pred_wavs = self.dcae.decode(latent.float(), sr=output_sr)
        wav = pred_wavs[0].cpu().float()
        wav_mono = wav.mean(dim=0)

        return output_sr, wav_mono
