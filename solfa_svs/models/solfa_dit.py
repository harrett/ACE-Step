"""
SolfaDiT: 12-layer Diffusion Transformer for DCAE latent space.

Reuses ACE-Step building blocks:
- PatchEmbed: Conv2d patchification (8, 16, L) → (B, L, D)
- LinearTransformerBlock: Linear self-attn + standard cross-attn + GLUMBConv FFN
- T2IFinalLayer: AdaLN + unpatchify back to (8, 16, L)
- Qwen2RotaryEmbedding: RoPE positional encoding
- Timesteps/TimestepEmbedding: Diffusion timestep conditioning

~92M parameters (12 blocks × ~7.3M + embeddings ~3.4M)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.utils import is_torch_version

from acestep.models.attention import LinearTransformerBlock
from acestep.models.ace_step_transformer import (
    PatchEmbed,
    T2IFinalLayer,
    Qwen2RotaryEmbedding,
)


class SolfaDiT(ModelMixin, ConfigMixin):
    """
    12-layer diffusion transformer operating in ACE-Step's DCAE latent space.

    Input:  noisy latent (B, 8, 16, L) + conditioning (B, L, 512) + timestep (B,)
    Output: denoised prediction (B, 8, 16, L)

    Architecture mirrors ACEStepTransformer2DModel.decode() but with smaller dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        mlp_ratio: float = 4.0,
        max_position: int = 8192,
        rope_theta: float = 10000.0,
        patch_size: List[int] = [16, 1],
        max_height: int = 16,
        max_width: int = 4096,
        conditioning_dim: int = 512,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.speaker_embedding_dim = speaker_embedding_dim

        # Speaker embedding projection
        # Dual injection: (1) AdaLN path for global timbre modulation,
        # (2) cross-attention token for fine-grained conditioning.
        # AdaLN injection is the primary path — it modulates every layer
        # and cannot be ignored by the model (unlike a single cross-attn token
        # among thousands of MIDI tokens).
        if speaker_embedding_dim > 0:
            self.speaker_embedder = nn.Linear(speaker_embedding_dim, conditioning_dim)
            self.speaker_adaln = nn.Sequential(
                nn.Linear(speaker_embedding_dim, inner_dim),
                nn.SiLU(),
                nn.Linear(inner_dim, inner_dim),
            )

        # RoPE positional encoding
        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=attention_head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
        )

        # Patchification: (B, 8, 16, L) → (B, L, inner_dim)
        self.proj_in = PatchEmbed(
            height=max_height,
            width=max_width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            bias=True,
        )

        # Transformer blocks with cross-attention
        self.transformer_blocks = nn.ModuleList([
            LinearTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                add_cross_attention=True,
                add_cross_attention_dim=conditioning_dim,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

        # Timestep embedding
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=inner_dim
        )
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(inner_dim, 6 * inner_dim, bias=True),
        )

        # Output: unpatchify (B, L, inner_dim) → (B, 8, 16, L)
        self.final_layer = T2IFinalLayer(
            hidden_size=inner_dim,
            patch_size=patch_size,
            out_channels=out_channels,
        )

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        hidden_states: torch.Tensor,                    # (B, 8, 16, L) noisy latent
        attention_mask: torch.Tensor,                    # (B, L)
        encoder_hidden_states: torch.Tensor,             # (B, L_cond, D_cond)
        encoder_hidden_mask: torch.Tensor,               # (B, L_cond)
        timestep: torch.Tensor,                          # (B,) diffusion timestep
        speaker_embeds: Optional[torch.Tensor] = None,   # (B, speaker_dim) or None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion transformer.

        Args:
            speaker_embeds: Optional speaker embedding. When provided and
                speaker_embedding_dim > 0, projects to a single token and
                prepends to encoder_hidden_states (following ACE-Step pattern).
                When None, behavior is identical to the original model.

        Returns: (B, 8, 16, L) predicted clean latent
        """
        output_length = hidden_states.shape[-1]
        B = hidden_states.shape[0]

        # Prepend speaker token to cross-attention context if available
        if self.speaker_embedding_dim > 0 and speaker_embeds is not None:
            # Cross-attention token (fine-grained): (B, speaker_dim) -> (B, 1, conditioning_dim)
            speaker_token = self.speaker_embedder(speaker_embeds).unsqueeze(1)
            speaker_mask = torch.ones(B, 1, device=hidden_states.device,
                                      dtype=encoder_hidden_mask.dtype)
            encoder_hidden_states = torch.cat(
                [speaker_token, encoder_hidden_states], dim=1
            )
            encoder_hidden_mask = torch.cat(
                [speaker_mask, encoder_hidden_mask], dim=1
            )

        # Timestep embedding → AdaLN modulation vector
        embedded_timestep = self.timestep_embedder(
            self.time_proj(timestep).to(dtype=hidden_states.dtype)
        )

        # AdaLN injection: add speaker to timestep embedding BEFORE t_block
        # so every transformer layer is modulated by speaker identity
        if self.speaker_embedding_dim > 0 and speaker_embeds is not None:
            speaker_adaln = self.speaker_adaln(speaker_embeds)  # (B, inner_dim)
            embedded_timestep = embedded_timestep + speaker_adaln

        temb = self.t_block(embedded_timestep)  # (B, 6 * inner_dim)

        # Patchify: (B, 8, 16, L) → (B, L, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # RoPE frequencies
        rotary_freqs_cis = self.rotary_emb(
            hidden_states, seq_len=hidden_states.shape[1]
        )
        encoder_rotary_freqs_cis = self.rotary_emb(
            encoder_hidden_states, seq_len=encoder_hidden_states.shape[1]
        )

        # Transformer blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_hidden_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                    temb=temb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_hidden_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                    temb=temb,
                )

        # Unpatchify: (B, L, inner_dim) → (B, 8, 16, L)
        output = self.final_layer(hidden_states, embedded_timestep, output_length)

        return output
