#!/usr/bin/env python
"""
Entry point for SolfaSVS inference.

Usage:
    python -m solfa_svs.scripts.infer \
        --checkpoint_path exps/solfa_svs/logs/solfa_svs/checkpoints/last.ckpt \
        --midi_path data/midi_for_ace/1084222.mid \
        --output_path outputs/generated.wav \
        --dcae_checkpoint_dir .cache/ace-step/checkpoints \
        --device cuda:0 --bf16

The MIDI files should be in midi_for_ace/ format with embedded sol-fa lyric
events (Chinese pinyin: Dao, Rei, Mi, Fa, So, La, Xi). An optional companion
.json metadata file can be placed alongside the .mid file.

Batch mode (generate from all .mid files in a directory):
    python -m solfa_svs.scripts.infer \
        --checkpoint_path exps/solfa_svs/logs/solfa_svs/checkpoints/last.ckpt \
        --midi_dir data/midi_for_ace \
        --output_path outputs/ \
        --dcae_checkpoint_dir .cache/ace-step/checkpoints \
        --device cuda:0 --bf16
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchaudio

from solfa_svs.models.pipeline import SolfaSVSPipeline


def generate_one(pipeline, midi_path, metadata_path, output_path, args):
    """Generate audio from a single MIDI file."""
    # Auto-detect companion metadata JSON if not specified
    if metadata_path is None:
        candidate = midi_path.replace(".mid", ".json")
        if os.path.exists(candidate):
            metadata_path = candidate

    print(f"Generating from MIDI: {midi_path}")
    if metadata_path:
        print(f"  Metadata: {metadata_path}")
    if args.reference_audio:
        print(f"  Reference audio: {args.reference_audio}")

    sr, wav = pipeline.generate_from_midi(
        midi_path=midi_path,
        metadata_path=metadata_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        omega_scale=args.omega_scale,
        output_sr=args.output_sr,
        seed=args.seed,
        apply_expression=not args.no_expression,
        reference_audio=args.reference_audio,
    )

    # Apply post-generation voice conversion if VC model provided
    if args.vc_model_path:
        print(f"  Applying voice conversion ({args.vc_backend})...")
        from solfa_svs.voice_conversion.vc_wrapper import VoiceConverter
        vc = VoiceConverter(
            model_path=args.vc_model_path,
            backend=args.vc_backend,
            index_path=args.vc_index_path,
            device=args.device,
        )
        sr, wav_np = vc.convert(
            input_audio=wav.numpy(),
            input_sr=sr,
            output_sr=sr,
        )
        wav = torch.from_numpy(wav_np).float()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torchaudio.save(
        output_path,
        wav.unsqueeze(0),  # (1, T)
        sample_rate=sr,
    )
    print(f"  Saved: {output_path} ({len(wav)/sr:.2f}s at {sr}Hz)")


def main():
    parser = argparse.ArgumentParser(description="SolfaSVS Inference")

    # Input — one of these is required
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--midi_path", type=str,
        help="Path to a single MIDI file (.mid) with sol-fa lyrics",
    )
    input_group.add_argument(
        "--midi_dir", type=str,
        help="Path to directory of MIDI files (batch mode)",
    )

    # Model
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to training checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs/generated.wav",
        help="Output WAV file path (or directory for batch mode)",
    )

    # MIDI metadata
    parser.add_argument(
        "--metadata_path", type=str, default=None,
        help="Path to companion .json metadata file for the MIDI "
             "(auto-detected if next to .mid file)",
    )

    # DCAE
    parser.add_argument(
        "--dcae_checkpoint_dir", type=str, default=None,
        help="ACE-Step checkpoint directory for DCAE (None = auto-download)",
    )

    # Generation params
    parser.add_argument("--num_inference_steps", type=int, default=60)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--omega_scale", type=float, default=10.0)
    parser.add_argument("--output_sr", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_expression", action="store_true",
                        help="Disable vibrato/portamento")

    # Speaker conditioning (Phase 2)
    parser.add_argument("--reference_audio", type=str, default=None,
                        help="Path to reference audio for zero-shot speaker cloning")
    parser.add_argument("--speaker_projection_path", type=str, default=None,
                        help="Path to speaker_projection.pt from preprocessing "
                             "(required for correct speaker embeddings)")

    # Voice conversion (Phase 1)
    parser.add_argument("--vc_model_path", type=str, default=None,
                        help="Path to VC model for post-generation timbre conversion")
    parser.add_argument("--vc_backend", type=str, default="rvc",
                        choices=["rvc", "sovits"],
                        help="Voice conversion backend (default: rvc)")
    parser.add_argument("--vc_index_path", type=str, default=None,
                        help="Path to RVC .index file for retrieval")

    # Device
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")

    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print("Loading pipeline...")
    pipeline = SolfaSVSPipeline.from_pretrained(
        checkpoint_path=args.checkpoint_path,
        dcae_checkpoint_dir=args.dcae_checkpoint_dir,
        speaker_projection_path=args.speaker_projection_path,
        device=args.device,
        dtype=dtype,
    )

    if args.midi_path:
        # Single file mode
        generate_one(pipeline, args.midi_path, args.metadata_path,
                     args.output_path, args)
    else:
        # Batch mode: process all .mid files in directory
        midi_dir = args.midi_dir
        output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)

        midi_files = sorted(
            f for f in os.listdir(midi_dir) if f.endswith(".mid")
        )
        print(f"Found {len(midi_files)} MIDI files in {midi_dir}")

        for i, fname in enumerate(midi_files):
            midi_path = os.path.join(midi_dir, fname)
            out_name = fname.replace(".mid", ".wav")
            output_path = os.path.join(output_dir, out_name)

            print(f"\n[{i+1}/{len(midi_files)}]")
            generate_one(pipeline, midi_path, None, output_path, args)

        print(f"\nDone. Generated {len(midi_files)} files in {output_dir}")


if __name__ == "__main__":
    main()
