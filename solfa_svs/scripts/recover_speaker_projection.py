#!/usr/bin/env python
"""
Recover the SpeakerEncoder projection weights from training data.

During preprocessing, SpeakerEncoder used a random Linear(192, 256) projection
to map CAM++ embeddings into the training space. Those weights were not saved.
This script recovers them by:
  1. Re-extracting raw CAM++ embeddings (192-dim) from the same audio files
  2. Loading the projected embeddings (256-dim) from existing .pt files
  3. Solving for the Linear weights via least squares

Usage (run on the server where funasr is installed):
    python -m solfa_svs.scripts.recover_speaker_projection \
        --latent_dir data/dcae_latents_spk \
        --audio_dirs ace_studio_exports=data/processed/ace_studio/audio \
                     Elirah=/data1/music/ACE-Step/data/processed/Elirah/audio \
                     Rowly=/data1/music/ACE-Step/data/processed/Rowly/audio \
        --samples_per_speaker 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def load_cam_encoder():
    """Load the frozen CAM++ speaker encoder."""
    from funasr import AutoModel
    return AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        disable_update=True,
    )


def extract_raw_embedding(encoder, audio_path: str) -> torch.Tensor:
    """Extract raw 192-dim CAM++ embedding from an audio file."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Truncate to 30s
    max_samples = 30 * 16000
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    wav_np = waveform.numpy()
    result = encoder.generate(input=wav_np, output_dir=None)
    if isinstance(result, list) and len(result) > 0:
        emb = result[0].get("spk_embedding", result[0].get("embedding"))
    else:
        emb = result

    if isinstance(emb, np.ndarray):
        return torch.from_numpy(emb).float().squeeze().cpu()
    elif isinstance(emb, torch.Tensor):
        return emb.float().squeeze().cpu()
    elif isinstance(emb, list):
        return torch.tensor(emb, dtype=torch.float32).cpu()
    else:
        raise ValueError(f"Unexpected embedding type: {type(emb)}")


def main():
    parser = argparse.ArgumentParser(
        description="Recover speaker projection weights from training data"
    )
    parser.add_argument(
        "--latent_dir", type=str, required=True,
        help="Directory with .pt files and train.json"
    )
    parser.add_argument(
        "--audio_dirs", type=str, nargs="+", required=True,
        help="speaker_name=audio_dir pairs, e.g. Elirah=/path/to/audio"
    )
    parser.add_argument(
        "--samples_per_speaker", type=int, default=50,
        help="Number of samples per speaker for least-squares fit (default: 50)"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output path for speaker_projection.pt (default: latent_dir/speaker_projection.pt)"
    )
    args = parser.parse_args()

    # Parse audio_dirs
    audio_base_paths = {}
    for pair in args.audio_dirs:
        name, path = pair.split("=", 1)
        audio_base_paths[name] = path
    print(f"Audio directories: {audio_base_paths}")

    # Load metadata
    with open(os.path.join(args.latent_dir, "train.json")) as f:
        meta = json.load(f)
    print(f"Total training samples: {len(meta)}")

    # Load CAM++ encoder
    print("Loading CAM++ speaker encoder...")
    encoder = load_cam_encoder()

    # Gather paired (raw_192, projected_256) from training data
    raw_embs = []
    proj_embs = []
    count = {name: 0 for name in audio_base_paths}

    for entry in tqdm(meta, desc="Extracting pairs"):
        speaker = entry.get("speaker_name", "")
        if speaker not in count or count[speaker] >= args.samples_per_speaker:
            continue

        # Load projected embedding from .pt
        pt_path = os.path.join(args.latent_dir, entry["pt_path"])
        if not os.path.exists(pt_path):
            continue
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        proj_emb = sample.get("speaker_embedding")
        if proj_emb is None or proj_emb.numel() == 0:
            continue

        # Find audio file
        audio_dir = audio_base_paths.get(speaker, "")
        audio_path = os.path.join(audio_dir, f"{entry['sample_id']}.wav")
        if not os.path.exists(audio_path):
            continue

        try:
            raw_emb = extract_raw_embedding(encoder, audio_path)
            if raw_emb.shape[0] != 192:
                print(f"  Unexpected raw dim: {raw_emb.shape}")
                continue
            raw_embs.append(raw_emb.cpu())
            proj_embs.append(proj_emb.cpu())
            count[speaker] += 1
        except Exception as e:
            print(f"  Error on {entry['sample_id']}: {e}")
            continue

        # Early exit if all speakers have enough samples
        if all(c >= args.samples_per_speaker for c in count.values()):
            break

    print(f"\nCollected {len(raw_embs)} paired samples: {count}")

    if len(raw_embs) < 10:
        print("ERROR: Not enough paired samples to recover projection!")
        sys.exit(1)

    # Solve for W, b in: projected = raw @ W.T + b
    X = torch.stack(raw_embs)   # (N, 192)
    Y = torch.stack(proj_embs)  # (N, 256)

    # Add bias column: X_aug = [X, 1]
    ones = torch.ones(X.shape[0], 1)
    X_aug = torch.cat([X, ones], dim=1)  # (N, 193)

    # Least squares: Y = X_aug @ params  =>  params = pinv(X_aug) @ Y
    solution = torch.linalg.lstsq(X_aug, Y).solution  # (193, 256)
    W = solution[:192, :].T  # (256, 192) = Linear weight
    b = solution[192, :]     # (256,) = Linear bias

    # Validate reconstruction
    Y_pred = X @ W.T + b.unsqueeze(0)
    residual = (Y - Y_pred).norm() / Y.norm()
    cos_sims = torch.nn.functional.cosine_similarity(Y, Y_pred, dim=1)
    print(f"\nRecovery quality:")
    print(f"  Relative residual: {residual:.6f}")
    print(f"  Mean cosine similarity: {cos_sims.mean():.6f}")
    print(f"  Min cosine similarity:  {cos_sims.min():.6f}")

    if cos_sims.mean() < 0.99:
        print("  WARNING: Recovery quality is low. The projection may not be a "
              "simple linear transform, or the encoder produces different outputs "
              "than during preprocessing.")

    # Save as nn.Linear state_dict
    state_dict = {"weight": W, "bias": b}
    out_path = args.output_path or os.path.join(args.latent_dir, "speaker_projection.pt")
    torch.save(state_dict, out_path)
    print(f"\nSaved recovered projection to: {out_path}")
    print(f"Use with: --speaker_projection_path {out_path}")


if __name__ == "__main__":
    main()
