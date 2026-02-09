"""
MIDI Encoder: NoteEncoder + FrameEncoder + FusionLayer

Produces conditioning embeddings for the SolfaDiT diffusion transformer
from MIDI-derived note events and frame-level features (F0, energy, phonemes).

Output: (B, L, 512) conditioning for cross-attention in SolfaDiT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NoteEncoder(nn.Module):
    """
    Encodes note-level events into note embeddings.

    Input: batched note features (phoneme_id, pitch, velocity, duration, position)
    Output: (B, N_notes, embed_dim) note embeddings
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_phonemes: int = 14,
        phoneme_embed_dim: int = 128,
        max_pitch: int = 128,
        pitch_embed_dim: int = 128,
        max_velocity: int = 128,
        velocity_embed_dim: int = 64,
        num_transformer_layers: int = 2,
        num_transformer_heads: int = 4,
    ):
        super().__init__()

        self.phoneme_embedding = nn.Embedding(num_phonemes, phoneme_embed_dim)
        self.pitch_embedding = nn.Embedding(max_pitch, pitch_embed_dim)
        self.velocity_embedding = nn.Embedding(max_velocity, velocity_embed_dim)
        self.duration_proj = nn.Linear(1, 64)
        self.position_proj = nn.Linear(1, 64)

        # Total input: 128 + 128 + 64 + 64 + 64 = 448
        self.input_proj = nn.Linear(448, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_transformer_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

    def forward(
        self,
        note_phonemes: torch.LongTensor,     # (B, N)
        note_pitches: torch.LongTensor,       # (B, N)
        note_velocities: torch.LongTensor,    # (B, N)
        note_durations: torch.FloatTensor,    # (B, N) in seconds
        note_positions: torch.FloatTensor,    # (B, N) normalized onset
        note_mask: torch.FloatTensor,         # (B, N) 1=valid, 0=pad
    ) -> torch.Tensor:
        """Returns (B, N, embed_dim) note embeddings."""
        phoneme_emb = self.phoneme_embedding(note_phonemes)
        pitch_emb = self.pitch_embedding(note_pitches.clamp(0, 127))
        velocity_emb = self.velocity_embedding(note_velocities.clamp(0, 127))
        duration_feat = self.duration_proj(
            torch.log1p(note_durations).unsqueeze(-1)
        )
        position_feat = self.position_proj(note_positions.unsqueeze(-1))

        # Concatenate all features
        note_feat = torch.cat(
            [phoneme_emb, pitch_emb, velocity_emb, duration_feat, position_feat],
            dim=-1,
        )  # (B, N, 448)

        note_feat = self.input_proj(note_feat)  # (B, N, embed_dim)

        # Transformer with padding mask (True = ignore)
        src_key_padding_mask = note_mask == 0
        note_embeddings = self.transformer(
            note_feat, src_key_padding_mask=src_key_padding_mask
        )

        return note_embeddings  # (B, N, embed_dim)


class FrameEncoder(nn.Module):
    """
    Encodes per-frame continuous features into frame embeddings.

    Input: f0 (B, L), energy (B, L), phonemes (B, L)
    Output: (B, L, embed_dim) frame embeddings
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_phonemes: int = 14,
        num_conv_layers: int = 3,
        conv_kernel_size: int = 3,
    ):
        super().__init__()

        self.f0_proj = nn.Linear(1, 256)
        self.energy_proj = nn.Linear(1, 256)
        self.phoneme_embedding = nn.Embedding(num_phonemes, embed_dim)

        # Conv1d stack for temporal smoothing
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(
                    embed_dim, embed_dim,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                ),
                nn.GroupNorm(32, embed_dim),
                nn.SiLU(),
            ])
        self.conv_stack = nn.Sequential(*conv_layers)

    def forward(
        self,
        f0: torch.FloatTensor,        # (B, L) in Hz
        energy: torch.FloatTensor,     # (B, L)
        phonemes: torch.LongTensor,    # (B, L)
    ) -> torch.Tensor:
        """Returns (B, L, embed_dim) frame embeddings."""
        # Log-F0 relative to A4 (440 Hz)
        f0_feat = self.f0_proj(
            torch.log2(f0.clamp(min=1e-6) / 440.0).unsqueeze(-1)
        )  # (B, L, 256)

        energy_feat = self.energy_proj(energy.unsqueeze(-1))  # (B, L, 256)

        phoneme_feat = self.phoneme_embedding(phonemes)  # (B, L, embed_dim)

        # Combine: cat f0+energy and add phoneme embedding
        continuous_feat = torch.cat([f0_feat, energy_feat], dim=-1)  # (B, L, 512)
        frame_feat = continuous_feat + phoneme_feat  # (B, L, 512)

        # Conv1d stack (expects B, C, L)
        frame_feat = frame_feat.transpose(1, 2)  # (B, 512, L)
        frame_feat = self.conv_stack(frame_feat)
        frame_feat = frame_feat.transpose(1, 2)  # (B, L, 512)

        return frame_feat


class FusionLayer(nn.Module):
    """
    Fuses note-level and frame-level embeddings via cross-attention.

    Frame embeddings attend to note embeddings, producing the final
    conditioning for the diffusion transformer.

    Input: frame_emb (B, L, D), note_emb (B, N, D), note_mask (B, N)
    Output: (B, L, D) fused conditioning
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 4):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        frame_embeddings: torch.Tensor,   # (B, L, D)
        note_embeddings: torch.Tensor,     # (B, N, D)
        note_mask: torch.FloatTensor,      # (B, N) 1=valid, 0=pad
    ) -> torch.Tensor:
        """Returns (B, L, D) fused embeddings."""
        # Cross-attention: frames query notes
        key_padding_mask = note_mask == 0  # True = ignore
        attn_output, _ = self.cross_attn(
            query=frame_embeddings,
            key=note_embeddings,
            value=note_embeddings,
            key_padding_mask=key_padding_mask,
        )

        # Residual + norm
        fused = self.norm(frame_embeddings + attn_output)
        output = self.output_proj(fused)

        return output  # (B, L, D)


class MidiEncoder(nn.Module):
    """
    Complete MIDI encoder: NoteEncoder + FrameEncoder + FusionLayer.

    Produces conditioning embeddings for cross-attention in SolfaDiT.

    Total params: ~4.5M
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_phonemes: int = 14,
        phoneme_embed_dim: int = 128,
        max_pitch: int = 128,
        pitch_embed_dim: int = 128,
        max_velocity: int = 128,
        velocity_embed_dim: int = 64,
        note_transformer_layers: int = 2,
        note_transformer_heads: int = 4,
        frame_conv_layers: int = 3,
        frame_conv_kernel: int = 3,
        fusion_heads: int = 4,
    ):
        super().__init__()

        self.note_encoder = NoteEncoder(
            embed_dim=embed_dim,
            num_phonemes=num_phonemes,
            phoneme_embed_dim=phoneme_embed_dim,
            max_pitch=max_pitch,
            pitch_embed_dim=pitch_embed_dim,
            max_velocity=max_velocity,
            velocity_embed_dim=velocity_embed_dim,
            num_transformer_layers=note_transformer_layers,
            num_transformer_heads=note_transformer_heads,
        )

        self.frame_encoder = FrameEncoder(
            embed_dim=embed_dim,
            num_phonemes=num_phonemes,
            num_conv_layers=frame_conv_layers,
            conv_kernel_size=frame_conv_kernel,
        )

        self.fusion = FusionLayer(
            embed_dim=embed_dim,
            num_heads=fusion_heads,
        )

    def forward(
        self,
        note_phonemes: torch.LongTensor,     # (B, N)
        note_pitches: torch.LongTensor,       # (B, N)
        note_velocities: torch.LongTensor,    # (B, N)
        note_durations: torch.FloatTensor,    # (B, N)
        note_positions: torch.FloatTensor,    # (B, N)
        note_mask: torch.FloatTensor,         # (B, N)
        f0: torch.FloatTensor,                # (B, L)
        energy: torch.FloatTensor,            # (B, L)
        phonemes: torch.LongTensor,           # (B, L)
        attention_mask: torch.FloatTensor,    # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoder_hidden_states: (B, L, embed_dim) conditioning embeddings
            encoder_mask: (B, L) attention mask for conditioning
        """
        # Note-level encoding
        note_emb = self.note_encoder(
            note_phonemes, note_pitches, note_velocities,
            note_durations, note_positions, note_mask,
        )  # (B, N, D)

        # Frame-level encoding
        frame_emb = self.frame_encoder(f0, energy, phonemes)  # (B, L, D)

        # Fuse note + frame
        fused = self.fusion(frame_emb, note_emb, note_mask)  # (B, L, D)

        return fused, attention_mask
