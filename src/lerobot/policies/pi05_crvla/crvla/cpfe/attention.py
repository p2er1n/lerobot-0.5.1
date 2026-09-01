"""Attention layers used by CPE and CAM."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class SelfAttention(nn.Module):
    """Multi-head self-attention with explicit projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min)
        attention = self.dropout(F.softmax(scores, dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(batch, length, channels)
        return self.out_proj(output)


class CrossAttention(nn.Module):
    """Multi-head attention from query tokens to context tokens."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=bias)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_length, query_dim = query.shape
        if key_value.shape[0] != batch:
            raise ValueError("query and context batch dimensions must match")
        context_length = key_value.shape[1]
        q = self.q_proj(query).reshape(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(key_value)
            .reshape(batch, context_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(key_value)
            .reshape(batch, context_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min)
        attention = self.dropout(F.softmax(scores, dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(batch, query_length, query_dim)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    """Pre-normalized cross-attention and feed-forward block."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_context = nn.LayerNorm(kv_dim)
        self.cross_attn = CrossAttention(query_dim, kv_dim, num_heads, dropout)
        self.norm_ffn = nn.LayerNorm(query_dim)
        hidden_dim = int(query_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        query = query + self.cross_attn(self.norm_query(query), self.norm_context(key_value))
        return query + self.mlp(self.norm_ffn(query))
