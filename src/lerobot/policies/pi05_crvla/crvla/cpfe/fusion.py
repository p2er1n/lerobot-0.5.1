"""Compression-aware visual modulation modules."""

import torch
import torch.nn as nn

from .attention import CrossAttention


def _validate_tokens(visual_tokens: torch.Tensor, prior_tokens: torch.Tensor, embed_dim: int) -> None:
    if visual_tokens.ndim != 3 or prior_tokens.ndim != 3:
        raise ValueError("visual and prior tokens must have shape (B, N, D)")
    if visual_tokens.shape[0] != prior_tokens.shape[0]:
        raise ValueError("visual and prior token batches must match")
    if visual_tokens.shape[-1] != embed_dim or prior_tokens.shape[-1] != embed_dim:
        raise ValueError("visual and prior token widths must match embed_dim")


class FusionBlock(nn.Module):
    """One pre-normalized CAM cross-attention block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, embed_dim, num_heads, dropout)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, visual_tokens: torch.Tensor, shared_context: torch.Tensor) -> torch.Tensor:
        visual_tokens = visual_tokens + self.cross_attn(
            self.norm_query(visual_tokens),
            self.norm_context(shared_context),
        )
        return visual_tokens + self.ffn(self.norm_ffn(visual_tokens))


class CPFEFusion(nn.Module):
    """Compression-aware modulation (CAM) from CPE priors to visual tokens.

    All blocks attend to the shared context ``[CPE priors; original visual
    tokens]``. Equation (6) gates the refined branch before final normalization.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        residual: bool = True,
        residual_gate_init: float = -4.0,
        num_fusion_layers: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers is not None and num_fusion_layers is not None and num_layers != num_fusion_layers:
            raise ValueError("num_layers and num_fusion_layers disagree")
        resolved_layers = num_layers if num_layers is not None else num_fusion_layers
        resolved_layers = 1 if resolved_layers is None else resolved_layers
        if resolved_layers <= 0:
            raise ValueError("CAM requires at least one fusion layer")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = resolved_layers
        self.residual = residual
        self.fusion_layers = nn.ModuleList(
            [FusionBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(resolved_layers)]
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, vit_tokens: torch.Tensor, cpfe_tokens: torch.Tensor) -> torch.Tensor:
        _validate_tokens(vit_tokens, cpfe_tokens, self.embed_dim)
        shared_context = torch.cat([cpfe_tokens, vit_tokens], dim=1)
        refined = vit_tokens
        for layer in self.fusion_layers:
            refined = layer(refined, shared_context)
        if not self.residual:
            return refined
        return self.output_norm(vit_tokens + torch.sigmoid(self.residual_gate) * refined)

    def get_gate_value(self, cpfe_tokens: torch.Tensor | None = None) -> torch.Tensor:
        del cpfe_tokens
        return torch.sigmoid(self.residual_gate)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, residual={self.residual}"
        )


class CPFEViTWrapper(nn.Module):
    """Apply CPE and CAM to externally computed visual tokens."""

    def __init__(
        self,
        cpfe: nn.Module,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        residual: bool = True,
        residual_gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.cpfe = cpfe
        self.fusion = CPFEFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            residual=residual,
            residual_gate_init=residual_gate_init,
        )

    def forward(self, images: torch.Tensor, vit_tokens: torch.Tensor) -> torch.Tensor:
        return self.fusion(vit_tokens, self.cpfe(images))

    def get_cpfe_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.cpfe(images)


class CPFEModulation(nn.Module):
    """Legacy FiLM baseline retained for ablation compatibility."""

    def __init__(
        self,
        embed_dim: int = 2176,
        cpfe_num_tokens: int = 32,
        gate_type: str = "scalar",
        initial_gate_bias: float = -2.0,
        scale_limit: float = 0.1,
        use_film: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if gate_type not in {"scalar", "channel"}:
            raise ValueError("gate_type must be 'scalar' or 'channel'")
        self.embed_dim = embed_dim
        self.cpfe_num_tokens = cpfe_num_tokens
        self.gate_type = gate_type
        self.initial_gate_bias = initial_gate_bias
        self.scale_limit = scale_limit
        self.use_film = use_film
        hidden_dim = max(embed_dim // 4, 1)
        gate_dim = 1 if gate_type == "scalar" else embed_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gate_dim),
        )
        self.gate_bias = nn.Parameter(torch.tensor(float(initial_gate_bias)))
        if use_film:
            self.scale_mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)
            )
            self.shift_mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)
            )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vit_tokens: torch.Tensor, cpfe_tokens: torch.Tensor) -> torch.Tensor:
        _validate_tokens(vit_tokens, cpfe_tokens, self.embed_dim)
        pooled = cpfe_tokens.mean(dim=1)
        gate = torch.sigmoid(self.gate_mlp(pooled) + self.gate_bias).unsqueeze(1)
        if not self.use_film:
            return vit_tokens
        scale = torch.tanh(self.scale_mlp(pooled)).unsqueeze(1) * self.scale_limit
        shift = self.shift_mlp(pooled).unsqueeze(1)
        modulated = vit_tokens * (1.0 + scale) + shift
        return vit_tokens + gate * (modulated - vit_tokens)

    def get_gate_value(self, cpfe_tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_mlp(cpfe_tokens.mean(dim=1)) + self.gate_bias)


class SimpleGateModulation(nn.Module):
    """Legacy pooled-prior baseline retained for ablation compatibility."""

    def __init__(
        self,
        embed_dim: int = 2176,
        cpfe_num_tokens: int = 32,
        initial_gate_bias: float = -2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cpfe_num_tokens = cpfe_num_tokens
        self.initial_gate_bias = initial_gate_bias
        hidden_dim = max(embed_dim // 4, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate_bias = nn.Parameter(torch.tensor(float(initial_gate_bias)))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vit_tokens: torch.Tensor, cpfe_tokens: torch.Tensor) -> torch.Tensor:
        _validate_tokens(vit_tokens, cpfe_tokens, self.embed_dim)
        pooled = cpfe_tokens.mean(dim=1)
        gate = torch.sigmoid(self.gate_mlp(pooled) + self.gate_bias).unsqueeze(1)
        return vit_tokens + gate * self.delta_mlp(pooled).unsqueeze(1)

    def get_gate_value(self, cpfe_tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_mlp(cpfe_tokens.mean(dim=1)) + self.gate_bias)


CAM = CPFEFusion
CompressionAwareModulation = CPFEFusion
