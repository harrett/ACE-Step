#!/usr/bin/env python
"""
Entry point for SolfaSVS training.

Usage:
    python -m solfa_svs.scripts.train \
        --train_metadata data/dcae_latents/train.json \
        --val_metadata data/dcae_latents/val.json \
        --latent_dir data/dcae_latents \
        --max_steps 100000 \
        --batch_size 4 \
        --accumulate_grad_batches 4
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from solfa_svs.training.trainer import SolfaSVSTrainer

torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description="Train SolfaSVS")

    # Data
    parser.add_argument("--train_metadata", type=str, default="data/dcae_latents/train.json")
    parser.add_argument("--val_metadata", type=str, default="data/dcae_latents/val.json")
    parser.add_argument("--latent_dir", type=str, default="data/dcae_latents")
    parser.add_argument("--max_seq_length", type=int, default=None)

    # Training
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cfg_dropout", type=float, default=0.15)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Flow matching
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)

    # Logging
    parser.add_argument("--exp_name", type=str, default="solfa_svs")
    parser.add_argument("--exp_dir", type=str, default="./exps",
                        help="Base directory for all experiments")
    parser.add_argument("--every_n_steps", type=int, default=1000)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    # Model configs (use defaults from plan)
    solfa_dit_config = {
        "in_channels": 8,
        "out_channels": 8,
        "num_layers": 12,
        "num_attention_heads": 8,
        "attention_head_dim": 64,
        "mlp_ratio": 4.0,
        "patch_size": [16, 1],
        "max_height": 16,
        "max_width": 4096,
        "max_position": 8192,
        "conditioning_dim": 512,
    }

    midi_encoder_config = {
        "embed_dim": 512,
        "num_phonemes": 14,
        "phoneme_embed_dim": 128,
        "max_pitch": 128,
        "pitch_embed_dim": 128,
        "max_velocity": 128,
        "velocity_embed_dim": 64,
        "note_transformer_layers": 2,
        "note_transformer_heads": 4,
        "frame_conv_layers": 3,
        "frame_conv_kernel": 3,
        "fusion_heads": 4,
    }

    # Create training module
    model = SolfaSVSTrainer(
        solfa_dit_config=solfa_dit_config,
        midi_encoder_config=midi_encoder_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        cfg_dropout=args.cfg_dropout,
        shift=args.shift,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        train_metadata=args.train_metadata,
        val_metadata=args.val_metadata,
        latent_dir=args.latent_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
    )

    # Logger — TensorBoardLogger creates: {exp_dir}/{exp_name}/version_X/
    logger = TensorBoardLogger(
        save_dir=args.exp_dir,
        name=args.exp_name,
    )

    # Callbacks — dirpath=None lets Lightning place checkpoints inside
    # the logger's version directory: {exp_dir}/{exp_name}/version_X/checkpoints/
    checkpoint_callback = ModelCheckpoint(
        filename="step={step}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        every_n_train_steps=args.every_n_steps,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Auto-detect strategy: DDP for multi-GPU, auto for single-GPU
    num_gpus = torch.cuda.device_count()
    strategy = "ddp_find_unused_parameters_true" if num_gpus > 1 else "auto"

    # Trainer
    trainer = Trainer(
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        strategy=strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=None,
        val_check_interval=args.every_n_steps,
        limit_val_batches=20,
    )

    print(f"Experiment dir: {logger.log_dir}")
    print(f"Checkpoints:    {logger.log_dir}/checkpoints/")

    # Train
    trainer.fit(model, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
