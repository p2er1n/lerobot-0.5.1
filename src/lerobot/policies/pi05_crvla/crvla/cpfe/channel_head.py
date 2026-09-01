"""Training-only expert head for simulated channel metadata."""

import torch
import torch.nn as nn

from .config import ChannelParameterHeadConfig


class ChannelParameterHead(nn.Module):
    """Decode codec and continuous channel targets from CPE features.

    The bandwidth simulator supplies the supervision for this head. The head is
    intentionally separate from CPE so it can be removed after post-training.
    """

    def __init__(self, config: ChannelParameterHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.codec_classifier = (
            nn.Linear(config.hidden_dim, config.num_codecs) if config.num_codecs > 0 else None
        )
        self.continuous_regressor = (
            nn.Linear(config.hidden_dim, config.num_continuous_parameters)
            if config.num_continuous_parameters > 0
            else None
        )

    @staticmethod
    def pool(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError("channel features must have shape (B, N, D) or (B, D)")

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.pool(features)
        if pooled.shape[-1] != self.config.input_dim:
            raise ValueError("channel feature width does not match input_dim")
        hidden = self.trunk(pooled)
        outputs: dict[str, torch.Tensor] = {}
        if self.codec_classifier is not None:
            outputs["codec_logits"] = self.codec_classifier(hidden)
        if self.continuous_regressor is not None:
            outputs["continuous"] = self.continuous_regressor(hidden)
        return outputs
