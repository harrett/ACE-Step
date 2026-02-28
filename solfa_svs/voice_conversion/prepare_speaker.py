#!/usr/bin/env python
"""
Prepare a speaker model for voice conversion from reference audio.

Wraps RVC or SO-VITS-SVC training to create a speaker model from
a collection of reference audio clips.

Usage:
    python -m solfa_svs.voice_conversion.prepare_speaker \
        --audio_dir /path/to/singer_audio/ \
        --output_dir models/speakers/singer_a/ \
        --backend rvc \
        --device cuda:0

Prerequisites:
    For RVC:    pip install rvc-python
    For SOVITS: pip install so-vits-svc-fork
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_rvc_speaker(
    audio_dir: str,
    output_dir: str,
    speaker_name: str = "target",
    sample_rate: int = 40000,
    epochs: int = 100,
    batch_size: int = 8,
    device: str = "cpu",
    f0_method: str = "rmvpe",
):
    """
    Train an RVC model from reference audio files.

    Args:
        audio_dir: Directory containing WAV/MP3 files of the target speaker
        output_dir: Directory to save trained model and index
        speaker_name: Name for the speaker model
        sample_rate: Training sample rate (40000 for RVC v2)
        epochs: Number of training epochs
        batch_size: Training batch size
        device: Computation device
        f0_method: F0 extraction method
    """
    try:
        from rvc_python.train import RVCTraining
    except ImportError:
        raise ImportError(
            "RVC training requires rvc-python. Install with: pip install rvc-python"
        )

    os.makedirs(output_dir, exist_ok=True)

    trainer = RVCTraining(device=device)

    # Collect audio files
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
        audio_files.extend(Path(audio_dir).glob(ext))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files for speaker '{speaker_name}'")

    # Preprocess
    trainer.preprocess(
        input_path=str(audio_dir),
        sr=sample_rate,
        n_processes=4,
    )

    # Extract features
    trainer.extract_features(
        f0method=f0_method,
        hop_length=160,
    )

    # Train
    trainer.train(
        model_name=speaker_name,
        total_epoch=epochs,
        batch_size=batch_size,
        save_every_epoch=25,
    )

    # Save model to output_dir
    model_path = trainer.get_model_path(speaker_name)
    index_path = trainer.get_index_path(speaker_name)

    import shutil
    if model_path and os.path.exists(model_path):
        dest = os.path.join(output_dir, f"{speaker_name}.pth")
        shutil.copy2(model_path, dest)
        print(f"Model saved: {dest}")

    if index_path and os.path.exists(index_path):
        dest = os.path.join(output_dir, f"{speaker_name}.index")
        shutil.copy2(index_path, dest)
        print(f"Index saved: {dest}")


def prepare_sovits_speaker(
    audio_dir: str,
    output_dir: str,
    speaker_name: str = "target",
    epochs: int = 100,
    batch_size: int = 8,
    device: str = "cpu",
):
    """
    Train a SO-VITS-SVC model from reference audio files.

    Args:
        audio_dir: Directory containing WAV files of the target speaker
        output_dir: Directory to save trained model
        speaker_name: Name for the speaker
        epochs: Number of training epochs
        batch_size: Training batch size
        device: Computation device
    """
    try:
        from so_vits_svc_fork.preprocessing.preprocess import preprocess as sovits_preprocess
        from so_vits_svc_fork.train import train as sovits_train
    except ImportError:
        raise ImportError(
            "SO-VITS-SVC training requires so-vits-svc-fork. "
            "Install with: pip install so-vits-svc-fork"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Organize audio into speaker directory
    speaker_dir = os.path.join(output_dir, "dataset_raw", speaker_name)
    os.makedirs(speaker_dir, exist_ok=True)

    # Symlink audio files
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        audio_files.extend(Path(audio_dir).glob(ext))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    for f in audio_files:
        dest = os.path.join(speaker_dir, f.name)
        if not os.path.exists(dest):
            os.symlink(str(f), dest)

    print(f"Found {len(audio_files)} audio files for speaker '{speaker_name}'")

    # Preprocess
    sovits_preprocess(
        input_dir=os.path.join(output_dir, "dataset_raw"),
        output_dir=os.path.join(output_dir, "dataset"),
    )

    # Train
    sovits_train(
        config_path=os.path.join(output_dir, "configs", "config.json"),
        model_path=os.path.join(output_dir, "logs"),
        epochs=epochs,
        batch_size=batch_size,
    )

    print(f"Model saved to: {os.path.join(output_dir, 'logs')}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a VC speaker model from reference audio"
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True,
        help="Directory containing reference audio files (WAV/MP3)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for trained speaker model",
    )
    parser.add_argument(
        "--backend", type=str, default="rvc", choices=["rvc", "sovits"],
        help="VC backend to use (default: rvc)",
    )
    parser.add_argument(
        "--speaker_name", type=str, default="target",
        help="Name for the speaker model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Computation device",
    )
    parser.add_argument(
        "--f0_method", type=str, default="rmvpe",
        help="F0 extraction method for RVC (rmvpe, crepe, harvest)",
    )

    args = parser.parse_args()

    if args.backend == "rvc":
        prepare_rvc_speaker(
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            speaker_name=args.speaker_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            f0_method=args.f0_method,
        )
    else:
        prepare_sovits_speaker(
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            speaker_name=args.speaker_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
