#!/usr/bin/env python
"""
Entry point for data preprocessing.

Usage:
    python -m solfa_svs.scripts.preprocess \
        --audio_dir data/processed/ace_studio/audio \
        --feature_dir data/processed/ace_studio_mel/mel_features \
        --output_dir data/dcae_latents \
        --train_metadata data/processed/ace_studio_mel/train_metadata.json \
        --val_metadata data/processed/ace_studio_mel/val_metadata.json \
        --device cuda:0
"""

import argparse
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solfa_svs.data.preprocess import run_preprocessing


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio to DCAE latents")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/processed/ace_studio/audio",
        help="Directory containing source WAV files",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="data/processed/ace_studio_mel/mel_features",
        help="Directory containing NPZ feature files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/dcae_latents",
        help="Output directory for .pt latent files",
    )
    parser.add_argument(
        "--train_metadata",
        type=str,
        default="data/processed/ace_studio_mel/train_metadata.json",
        help="Path to train metadata JSON",
    )
    parser.add_argument(
        "--val_metadata",
        type=str,
        default="data/processed/ace_studio_mel/val_metadata.json",
        help="Path to val metadata JSON",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="ACE-Step checkpoint directory (None = auto-download from HuggingFace)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (cpu, cuda:0, mps)",
    )
    parser.add_argument(
        "--validate_n",
        type=int,
        default=5,
        help="Number of samples for round-trip DCAE validation",
    )

    args = parser.parse_args()

    run_preprocessing(
        audio_dir=args.audio_dir,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        train_metadata_path=args.train_metadata,
        val_metadata_path=args.val_metadata,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        validate_n=args.validate_n,
    )


if __name__ == "__main__":
    main()
