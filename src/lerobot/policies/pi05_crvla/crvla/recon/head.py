"""Residual correction head for intermediate VLM visual features."""

import torch
import torch.nn as nn

from .config import ReconHeadConfig


class SpatialMixerBlock(nn.Module):
    """Mix neighboring patches independently within each camera view."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.GroupNorm(num_groups, dim)
        self.act = nn.SiLU()
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pointwise(x)
        return residual + self.dropout(x)


class VLMFeatureReconstructionHead(nn.Module):
    """Predict ``R_theta(H_comp)`` and return ``H_comp + R_theta(H_comp)``."""

    def __init__(self, config: ReconHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.bottleneck_dim = config.bottleneck_dim
        self.patch_grid_size = config.patch_grid_size
        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.in_proj = nn.Linear(config.embed_dim, config.bottleneck_dim)
        self.mixer_blocks = nn.ModuleList(
            [
                SpatialMixerBlock(
                    dim=config.bottleneck_dim,
                    kernel_size=config.conv_kernel_size,
                    num_groups=config.num_groups,
                    dropout=config.dropout,
                )
                for _ in range(config.num_mixer_blocks)
            ]
        )
        self.act = nn.GELU()
        self.out_proj = nn.Linear(config.bottleneck_dim, config.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Start as an exact identity correction.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _infer_num_views(self, num_tokens: int) -> int:
        patches_per_view = self.config.num_patches_per_view
        if num_tokens == 0 or num_tokens % patches_per_view != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a positive multiple of "
                f"patches_per_view ({patches_per_view})"
            )
        return num_tokens // patches_per_view

    def predict_residual(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        if vision_tokens.ndim != 3:
            raise ValueError("vision_tokens must have shape (B, N, D)")
        batch, num_tokens, width = vision_tokens.shape
        if width != self.embed_dim:
            raise ValueError("vision token width does not match embed_dim")

        num_views = self._infer_num_views(num_tokens)
        grid = self.patch_grid_size
        x = self.in_proj(self.input_norm(vision_tokens))
        x = x.reshape(batch * num_views, grid, grid, self.bottleneck_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        for block in self.mixer_blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(batch, num_tokens, self.bottleneck_dim)
        return self.out_proj(self.act(x))

    def forward_with_residual(self, vision_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.predict_residual(vision_tokens)
        return vision_tokens + residual, residual

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        reconstructed, _ = self.forward_with_residual(vision_tokens)
        return reconstructed

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, bottleneck_dim={self.bottleneck_dim}, "
            f"patch_grid_size={self.patch_grid_size}, layer_idx={self.config.layer_idx}"
        )
