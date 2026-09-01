"""Compression-prior Action Anchor adapted to the PI0.5 action expert.

The attention layout follows the MIT-licensed CR-VLA reference implementation.
PI0.5 uses different widths for its VLM and action expert, so the context key/value
projections perform the required width conversion directly.
"""

import math

import torch
from torch import nn


class CompressionPriorActionAnchorBlock(nn.Module):
    """Retrieve action, VLM, and compression-prior context for action tokens."""

    def __init__(
        self,
        action_dim: int,
        vlm_dim: int,
        prior_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim % num_heads != 0:
            raise ValueError("action_dim must be divisible by num_heads")
        self.action_dim = action_dim
        self.vlm_dim = vlm_dim
        self.prior_dim = prior_dim
        self.num_heads = num_heads
        self.head_dim = action_dim // num_heads

        self.query_norm = nn.LayerNorm(action_dim)
        self.query_projection = nn.Linear(action_dim, action_dim)
        self.k_self = nn.Linear(action_dim, action_dim)
        self.v_self = nn.Linear(action_dim, action_dim)
        self.k_vlm = nn.Linear(vlm_dim, action_dim)
        self.v_vlm = nn.Linear(vlm_dim, action_dim)
        self.k_prior = nn.Linear(prior_dim, action_dim)
        self.v_prior = nn.Linear(prior_dim, action_dim)

        self.prior_gate = nn.Sequential(
            nn.Linear(action_dim + prior_dim, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.output_projection = nn.Linear(3 * action_dim, action_dim)
        self.attention_dropout = nn.Dropout(dropout)

        ffn_dim = int(ffn_ratio * action_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, action_dim),
        )
        self._init_identity_residual()

    def _init_identity_residual(self) -> None:
        # A copied PI0.5 checkpoint must initially preserve its action embeddings.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.action_dim)

    def _attend(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_projection: nn.Linear,
        value_projection: nn.Linear,
    ) -> torch.Tensor:
        keys = self._split_heads(key_projection(context))
        values = self._split_heads(value_projection(context))
        weights = torch.softmax(query @ keys.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        return self._merge_heads(self.attention_dropout(weights) @ values)

    def forward(
        self,
        action_queries: torch.Tensor,
        vlm_tokens: torch.Tensor,
        compression_prior: torch.Tensor,
    ) -> torch.Tensor:
        if action_queries.ndim != 3 or vlm_tokens.ndim != 3 or compression_prior.ndim != 3:
            raise ValueError("Action Anchor inputs must be rank-3 tensors")
        if not (action_queries.shape[0] == vlm_tokens.shape[0] == compression_prior.shape[0]):
            raise ValueError("Action Anchor inputs must share a batch size")
        if action_queries.shape[-1] != self.action_dim:
            raise ValueError("action query width does not match action_dim")
        if vlm_tokens.shape[-1] != self.vlm_dim:
            raise ValueError("VLM token width does not match vlm_dim")
        if compression_prior.shape[-1] != self.prior_dim:
            raise ValueError("compression-prior width does not match prior_dim")

        query = self._split_heads(self.query_projection(self.query_norm(action_queries)))
        self_output = self._attend(query, action_queries, self.k_self, self.v_self)
        vlm_output = self._attend(query, vlm_tokens, self.k_vlm, self.v_vlm)
        prior_output = self._attend(query, compression_prior, self.k_prior, self.v_prior)

        gate_input = torch.cat(
            [action_queries.mean(dim=1), compression_prior.mean(dim=1)],
            dim=-1,
        )
        prior_gate = torch.tanh(self.alpha) * torch.sigmoid(self.prior_gate(gate_input))
        anchored_prior_output = prior_gate[:, None] * prior_output

        fused = self.output_projection(torch.cat([self_output, vlm_output, anchored_prior_output], dim=-1))
        hidden = action_queries + fused
        return hidden + self.ffn(hidden)


class CompressionPriorActionAnchor(nn.Module):
    """Stack one or more Action Anchor blocks."""

    def __init__(
        self,
        action_dim: int,
        vlm_dim: int,
        prior_dim: int,
        num_layers: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("Action Anchor requires at least one layer")
        self.layers = nn.ModuleList(
            [
                CompressionPriorActionAnchorBlock(
                    action_dim=action_dim,
                    vlm_dim=vlm_dim,
                    prior_dim=prior_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        action_queries: torch.Tensor,
        vlm_tokens: torch.Tensor,
        compression_prior: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            action_queries = layer(action_queries, vlm_tokens, compression_prior)
        return action_queries
