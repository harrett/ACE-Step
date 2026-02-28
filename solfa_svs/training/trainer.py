"""
Training module for SolfaSVS.

PyTorch Lightning module implementing flow-matching diffusion training
for the SolfaDiT + MidiEncoder models.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from typing import Optional

from solfa_svs.models.solfa_dit import SolfaDiT
from solfa_svs.models.midi_encoder import MidiEncoder
from solfa_svs.data.dataset import SolfaDataset, collate_fn
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class SolfaSVSTrainer(LightningModule):
    """
    PyTorch Lightning training module for SolfaSVS.

    Implements flow-matching diffusion training following ACE-Step's approach:
    - Logit-normal timestep sampling
    - Flow-matching noise schedule with shift
    - Preconditioning on model predictions
    - Classifier-free guidance dropout
    - Optional speaker conditioning with separate CFG dropout rate
    """

    def __init__(
        self,
        # Model config
        solfa_dit_config: Optional[dict] = None,
        midi_encoder_config: Optional[dict] = None,
        # Training config
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.95),
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        gradient_clip_val: float = 0.5,
        gradient_checkpointing: bool = True,
        cfg_dropout: float = 0.15,
        # Speaker config
        speaker_dim: int = 0,
        speaker_cfg_dropout: float = 0.5,
        speaker_warmup: bool = False,
        # Flow matching config
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        # Data config
        train_metadata: str = "data/dcae_latents/train.json",
        val_metadata: str = "data/dcae_latents/val.json",
        latent_dir: str = "data/dcae_latents",
        batch_size: int = 4,
        num_workers: int = 4,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Default configs
        if solfa_dit_config is None:
            solfa_dit_config = {}
        if midi_encoder_config is None:
            midi_encoder_config = {}

        # Inject speaker_embedding_dim into SolfaDiT config
        if speaker_dim > 0:
            solfa_dit_config["speaker_embedding_dim"] = speaker_dim

        # Build models
        self.solfa_dit = SolfaDiT(**solfa_dit_config)
        self.midi_encoder = MidiEncoder(**midi_encoder_config)

        if gradient_checkpointing:
            self.solfa_dit.enable_gradient_checkpointing()

        # Scheduler for flow-matching
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

    def get_sigmas(self, timesteps: torch.Tensor, n_dim: int) -> torch.Tensor:
        """Get sigma values for given timesteps, expanded to n_dim dimensions."""
        sigmas = self.scheduler.sigmas.to(
            device=timesteps.device, dtype=torch.float32
        )
        schedule_timesteps = self.scheduler.timesteps.to(timesteps.device)

        step_indices = [
            (schedule_timesteps == t).nonzero().item() for t in timesteps
        ]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps using logit-normal distribution."""
        u = torch.normal(
            mean=self.hparams.logit_mean,
            std=self.hparams.logit_std,
            size=(batch_size,),
            device="cpu",
        )
        u = torch.sigmoid(u)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        indices = indices.clamp(0, self.scheduler.config.num_train_timesteps - 1)
        timesteps = self.scheduler.timesteps[indices].to(device)
        return timesteps

    def training_step(self, batch, batch_idx):
        target_latent = batch["latent"]            # (B, 8, 16, L)
        attention_mask = batch["attention_mask"]    # (B, L)
        device = target_latent.device
        dtype = target_latent.dtype
        B = target_latent.shape[0]

        # 1. MIDI encoder → conditioning
        encoder_hidden_states, encoder_mask = self.midi_encoder(
            note_phonemes=batch["note_phonemes"],
            note_pitches=batch["note_pitches"],
            note_velocities=batch["note_velocities"],
            note_durations=batch["note_durations"],
            note_positions=batch["note_positions"],
            note_mask=batch["note_mask"],
            f0=batch["f0"],
            energy=batch["energy"],
            phonemes=batch["phonemes"],
            attention_mask=attention_mask,
        )  # (B, L, 512), (B, L)

        # 2. CFG dropout: zero out MIDI conditioning with probability
        if self.training and self.hparams.cfg_dropout > 0:
            cfg_mask = (
                torch.rand(B, device=device) < self.hparams.cfg_dropout
            ).float()
            # Zero out conditioning for selected samples
            encoder_hidden_states = encoder_hidden_states * (
                1.0 - cfg_mask[:, None, None]
            )
            encoder_mask = encoder_mask * (1.0 - cfg_mask[:, None])

        # 2b. Speaker embedding with separate CFG dropout
        speaker_embeds = None
        if self.hparams.speaker_dim > 0:
            speaker_embeds = batch["speaker_embedding"].to(device=device, dtype=dtype)
            # Speaker CFG dropout: zero out speaker embedding independently
            if self.training and self.hparams.speaker_cfg_dropout > 0:
                spk_cfg_mask = (
                    torch.rand(B, device=device) < self.hparams.speaker_cfg_dropout
                ).float()
                speaker_embeds = speaker_embeds * (1.0 - spk_cfg_mask[:, None])

        # 3. Sample timesteps (logit-normal)
        timesteps = self.sample_timesteps(B, device)

        # 4. Get sigmas and create noisy latent
        sigmas = self.get_sigmas(timesteps, n_dim=target_latent.ndim)
        noise = torch.randn_like(target_latent)
        noisy_latent = sigmas * noise + (1.0 - sigmas) * target_latent

        # 5. Forward pass
        model_pred = self.solfa_dit(
            hidden_states=noisy_latent,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_mask,
            timestep=timesteps.to(dtype),
            speaker_embeds=speaker_embeds,
        )

        # 6. Preconditioning (Section 5 of https://arxiv.org/abs/2206.00364)
        model_pred = model_pred * (-sigmas) + noisy_latent

        # 7. Compute MSE loss with attention mask
        mask = (
            attention_mask.unsqueeze(1).unsqueeze(1)
            .expand(-1, target_latent.shape[1], target_latent.shape[2], -1)
        )

        selected_pred = (model_pred * mask).reshape(B, -1).contiguous()
        selected_target = (target_latent * mask).reshape(B, -1).contiguous()

        loss = F.mse_loss(selected_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(B, -1).mean(1)
        loss = loss.mean()

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Log learning rate (only when attached to a Trainer)
        try:
            if self.lr_schedulers() is not None:
                lr = self.lr_schedulers().get_last_lr()[0]
                self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        except RuntimeError:
            pass

        return loss

    def validation_step(self, batch, batch_idx):
        target_latent = batch["latent"]
        attention_mask = batch["attention_mask"]
        device = target_latent.device
        dtype = target_latent.dtype
        B = target_latent.shape[0]

        # MIDI encoder (no CFG dropout during validation)
        encoder_hidden_states, encoder_mask = self.midi_encoder(
            note_phonemes=batch["note_phonemes"],
            note_pitches=batch["note_pitches"],
            note_velocities=batch["note_velocities"],
            note_durations=batch["note_durations"],
            note_positions=batch["note_positions"],
            note_mask=batch["note_mask"],
            f0=batch["f0"],
            energy=batch["energy"],
            phonemes=batch["phonemes"],
            attention_mask=attention_mask,
        )

        # Speaker embedding (no dropout during validation)
        speaker_embeds = None
        if self.hparams.speaker_dim > 0:
            speaker_embeds = batch["speaker_embedding"].to(device=device, dtype=dtype)

        timesteps = self.sample_timesteps(B, device)
        sigmas = self.get_sigmas(timesteps, n_dim=target_latent.ndim)
        noise = torch.randn_like(target_latent)
        noisy_latent = sigmas * noise + (1.0 - sigmas) * target_latent

        model_pred = self.solfa_dit(
            hidden_states=noisy_latent,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_mask,
            timestep=timesteps.to(dtype),
            speaker_embeds=speaker_embeds,
        )

        model_pred = model_pred * (-sigmas) + noisy_latent

        mask = (
            attention_mask.unsqueeze(1).unsqueeze(1)
            .expand(-1, target_latent.shape[1], target_latent.shape[2], -1)
        )

        selected_pred = (model_pred * mask).reshape(B, -1).contiguous()
        selected_target = (target_latent * mask).reshape(B, -1).contiguous()

        loss = F.mse_loss(selected_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(B, -1).mean(1)
        loss = loss.mean()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )

        # Linear warmup + cosine decay
        def lr_lambda(step):
            warmup = self.hparams.warmup_steps
            max_steps = self.hparams.max_steps
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        dataset = SolfaDataset(
            metadata_path=self.hparams.train_metadata,
            latent_dir=self.hparams.latent_dir,
            max_length=self.hparams.max_seq_length,
            speaker_dim=self.hparams.speaker_dim,
        )
        nw = self.hparams.num_workers
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=nw > 0,
        )

    def val_dataloader(self):
        dataset = SolfaDataset(
            metadata_path=self.hparams.val_metadata,
            latent_dir=self.hparams.latent_dir,
            max_length=self.hparams.max_seq_length,
            speaker_dim=self.hparams.speaker_dim,
        )
        nw = self.hparams.num_workers
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=nw > 0,
        )
