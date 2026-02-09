"""
Data Preprocessing: Encode audio to DCAE latents and resample MIDI features.

Loads audio files, encodes through ACE-Step's frozen DCAE encoder,
resamples MIDI features to latent frame rate, and saves consolidated
training samples as .pt files.
"""

import os
import sys
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add project root to path for ACE-Step imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from solfa_svs.data.midi_parser import load_features_from_npz
from solfa_svs.data.f0_expression import add_expression


# DCAE latent frame rate: 44100 / 512 / 8
LATENT_FRAME_RATE = 44100 / 512 / 8  # ≈ 10.766601562
MEL_FRAME_RATE = 44100 / 512  # 44.1kHz / 512 hop ≈ 86.13 fps


def load_dcae(checkpoint_dir: Optional[str] = None, device: str = "cpu") -> MusicDCAE:
    """Load pretrained MusicDCAE from ACE-Step checkpoint."""
    if checkpoint_dir is not None:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        dcae_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_path = os.path.join(checkpoint_dir, "music_vocoder")
        dcae = MusicDCAE(
            dcae_checkpoint_path=dcae_path,
            vocoder_checkpoint_path=vocoder_path,
        )
    else:
        dcae = MusicDCAE()
    dcae = dcae.to(device).eval()
    dcae.requires_grad_(False)
    return dcae


def encode_audio(dcae: MusicDCAE, audio_path: str, device: str = "cpu") -> tuple:
    """
    Encode a single audio file to DCAE latent representation.

    Args:
        dcae: Loaded MusicDCAE model
        audio_path: Path to WAV file
        device: Device for computation

    Returns:
        latent: Tensor (8, 16, L)
        latent_length: int (valid latent frames)
    """
    audio, sr = torchaudio.load(audio_path)

    # DCAE expects stereo input (2, T)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    audio = audio.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, 2, T)

    with torch.no_grad():
        latents, latent_lengths = dcae.encode(audio, sr=sr)

    latent = latents[0].cpu()  # (8, 16, L)
    latent_length = latent_lengths[0].item()

    return latent, latent_length


def validate_roundtrip(
    dcae: MusicDCAE,
    audio_path: str,
    device: str = "cpu",
    target_sr: int = 44100,
) -> dict:
    """
    Encode → decode round-trip validation for a single audio file.

    Returns dict with SNR and reconstruction info.
    """
    # Load original
    audio_orig, sr_orig = torchaudio.load(audio_path)
    if audio_orig.shape[0] == 1:
        audio_orig = audio_orig.repeat(2, 1)

    audio_input = audio_orig.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents, latent_lengths = dcae.encode(audio_input, sr=sr_orig)
        _, pred_wavs = dcae.decode(latents, sr=target_sr)

    pred_wav = pred_wavs[0].cpu().float()  # (2, T')

    # Compute simple SNR on mono
    orig_mono = audio_orig.mean(dim=0)
    pred_mono = pred_wav.mean(dim=0)

    # Resample original to target_sr for comparison
    if sr_orig != target_sr:
        resampler = torchaudio.transforms.Resample(sr_orig, target_sr)
        orig_mono = resampler(orig_mono.unsqueeze(0)).squeeze(0)

    # Trim to same length
    min_len = min(len(orig_mono), len(pred_mono))
    orig_mono = orig_mono[:min_len]
    pred_mono = pred_mono[:min_len]

    # SNR
    noise = orig_mono - pred_mono
    signal_power = (orig_mono ** 2).mean()
    noise_power = (noise ** 2).mean()
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

    return {
        "snr_db": snr.item(),
        "orig_length": min_len,
        "latent_shape": list(latents.shape),
    }


def preprocess_sample(
    dcae: MusicDCAE,
    audio_path: str,
    npz_path: str,
    output_path: str,
    device: str = "cpu",
    apply_expression: bool = True,
) -> bool:
    """
    Preprocess a single sample: encode audio + resample features + save .pt

    Args:
        dcae: Loaded MusicDCAE
        audio_path: Path to WAV file
        npz_path: Path to NPZ with f0, phonemes, energy
        output_path: Path to save .pt file
        device: Computation device
        apply_expression: Whether to apply vibrato/portamento to F0

    Returns:
        True if successful
    """
    try:
        # 1. Encode audio to DCAE latent
        latent, latent_length = encode_audio(dcae, audio_path, device)

        # 2. Load and resample features to latent frame rate
        features = load_features_from_npz(
            npz_path,
            source_frame_rate=MEL_FRAME_RATE,
            target_frame_rate=LATENT_FRAME_RATE,
            target_length=latent_length,
        )

        # 3. Apply F0 expression (vibrato/portamento)
        f0 = features["f0"]
        phonemes = features["phonemes"]
        if apply_expression and len(f0) > 0:
            frame_period_ms = 1000.0 / LATENT_FRAME_RATE  # ~92.9ms
            f0 = add_expression(
                f0,
                phonemes=phonemes,
                frame_period_ms=frame_period_ms,
                # Adjusted for lower frame rate
                min_note_frames=3,    # ~280ms at 10.77fps
                onset_delay_frames=2,  # ~186ms
                ramp_in_frames=2,      # ~186ms
                transition_frames=1,   # ~93ms
            )

        # 4. Save consolidated sample
        sample = {
            "latent": latent,                                    # (8, 16, L)
            "latent_length": latent_length,                      # int
            "notes": features["notes"],                          # List[Dict]
            "f0": torch.from_numpy(f0).float(),                  # (L,)
            "energy": torch.from_numpy(features["energy"]).float(),  # (L,)
            "phonemes": torch.from_numpy(features["phonemes"]).long(),  # (L,)
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(sample, output_path)
        return True

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False


def run_preprocessing(
    audio_dir: str,
    feature_dir: str,
    output_dir: str,
    train_metadata_path: str,
    val_metadata_path: str,
    checkpoint_dir: Optional[str] = None,
    device: str = "cpu",
    validate_n: int = 5,
):
    """
    Run full preprocessing pipeline.

    Args:
        audio_dir: Directory containing WAV files
        feature_dir: Directory containing NPZ feature files
        output_dir: Output directory for .pt latent files
        train_metadata_path: Path to train_metadata.json
        val_metadata_path: Path to val_metadata.json
        checkpoint_dir: ACE-Step checkpoint directory (None = auto-download)
        device: Computation device
        validate_n: Number of samples for round-trip validation
    """
    print("Loading DCAE model...")
    dcae = load_dcae(checkpoint_dir, device)

    # Load metadata
    with open(train_metadata_path) as f:
        train_meta = json.load(f)
    with open(val_metadata_path) as f:
        val_meta = json.load(f)

    all_meta = train_meta + val_meta
    print(f"Total samples: {len(all_meta)} (train: {len(train_meta)}, val: {len(val_meta)})")

    # Round-trip validation on first N samples
    if validate_n > 0:
        print(f"\n=== Round-trip validation on {validate_n} samples ===")
        for i, sample in enumerate(all_meta[:validate_n]):
            audio_path = os.path.join(
                os.path.dirname(train_metadata_path),
                sample.get("audio_path", f"audio/{sample['sample_id']}.wav"),
            )
            if not os.path.exists(audio_path):
                # Try absolute path in audio_dir
                audio_path = os.path.join(audio_dir, f"{sample['sample_id']}.wav")

            if os.path.exists(audio_path):
                result = validate_roundtrip(dcae, audio_path, device)
                print(f"  [{i+1}] {sample['sample_id']}: SNR={result['snr_db']:.2f}dB, "
                      f"latent={result['latent_shape']}")
            else:
                print(f"  [{i+1}] {sample['sample_id']}: audio not found at {audio_path}")

    # Process all samples
    os.makedirs(output_dir, exist_ok=True)

    train_entries = []
    val_entries = []

    for split_name, split_meta, split_entries in [
        ("train", train_meta, train_entries),
        ("val", val_meta, val_entries),
    ]:
        print(f"\n=== Processing {split_name} split ({len(split_meta)} samples) ===")

        for sample in tqdm(split_meta, desc=split_name):
            sample_id = sample["sample_id"]

            # Resolve audio path
            audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
            if not os.path.exists(audio_path):
                base_dir = os.path.dirname(train_metadata_path)
                audio_path = os.path.join(base_dir, sample.get("audio_path", ""))

            # Resolve feature path
            npz_path = os.path.join(feature_dir, f"{sample_id}.npz")
            if not os.path.exists(npz_path):
                base_dir = os.path.dirname(train_metadata_path)
                npz_path = os.path.join(base_dir, sample.get("feature_path", ""))

            if not os.path.exists(audio_path):
                print(f"  Skip {sample_id}: audio not found")
                continue
            if not os.path.exists(npz_path):
                print(f"  Skip {sample_id}: features not found")
                continue

            output_path = os.path.join(output_dir, f"{sample_id}.pt")

            success = preprocess_sample(
                dcae, audio_path, npz_path, output_path, device,
            )

            if success:
                split_entries.append({
                    "sample_id": sample_id,
                    "pt_path": f"{sample_id}.pt",
                    "duration": sample.get("duration", 0),
                })

    # Save new metadata
    train_json_path = os.path.join(output_dir, "train.json")
    val_json_path = os.path.join(output_dir, "val.json")

    with open(train_json_path, "w") as f:
        json.dump(train_entries, f, indent=2)
    with open(val_json_path, "w") as f:
        json.dump(val_entries, f, indent=2)

    print(f"\nDone! Processed {len(train_entries)} train + {len(val_entries)} val samples")
    print(f"Saved to: {output_dir}")
