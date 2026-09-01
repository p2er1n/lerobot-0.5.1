"""Compression prior extraction from CaVE encoder features."""

import warnings
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from .attention import CrossAttention
from .cave import CaVE
from .config import CPFEConfig


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping of parameter tensors")
    tensor_items = {str(key): value for key, value in checkpoint.items() if torch.is_tensor(value)}
    if tensor_items:
        return tensor_items
    for key in (
        "params_ema",
        "params",
        "state_dict",
        "model_state_dict",
        "model",
        "module",
        "encoder",
        "cave",
    ):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            return _extract_state_dict(nested)
    raise ValueError("no tensor state dictionary was found in the checkpoint")


def _strip_checkpoint_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "cpfe.", "cpe.", "cave.", "cave_encoder.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


class CPFE(nn.Module):
    """Compression Prior Feature Extractor.

    CaVE features are projected to the visual-token width. Learned queries then
    aggregate the feature sequence through cross-attention to produce a compact
    set of compression-prior tokens.
    """

    def __init__(self, config: CPFEConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = CPFEConfig(**config)
        self.config = config
        self.cave = CaVE(
            in_nc=config.cave_in_nc,
            out_nc=config.cave_out_nc,
            nc=config.cave_nc,
            nb=config.cave_nb,
            act_mode=config.cave_act_mode,
        )
        if config.cave_checkpoint:
            self._load_cave_checkpoint(config.cave_checkpoint, config.strict_cave_checkpoint)

        self.input_proj = nn.Sequential(
            nn.Linear(config.cave_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.prior_tokens = nn.Parameter(torch.empty(1, config.num_prior_tokens, config.hidden_dim))
        self.pos_embed = (
            nn.Parameter(torch.empty(1, config.max_seq_len, config.hidden_dim))
            if config.use_pos_embed
            else None
        )
        self.query_attention = nn.ModuleList(
            [
                CrossAttention(
                    query_dim=config.hidden_dim,
                    kv_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_attention_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self._init_trainable_weights()

        if config.freeze_cave:
            self._freeze_cave()

    @property
    def cave_encoder(self) -> nn.Module:
        """Expose the CaVE module that contains the published encoder path."""
        return self.cave

    def _load_cave_checkpoint(self, checkpoint_path: str, strict: bool) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            message = f"CaVE checkpoint does not exist: {path}"
            if strict:
                raise FileNotFoundError(message)
            warnings.warn(message, stacklevel=2)
            return
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            warnings.warn(
                "This PyTorch version does not support weights_only checkpoint loading.",
                stacklevel=2,
            )
            checkpoint = torch.load(path, map_location="cpu")

        source = _extract_state_dict(checkpoint)
        target = self.cave.state_dict()
        compatible = {}
        for source_key, value in source.items():
            key = _strip_checkpoint_prefixes(source_key)
            candidates = (key, f"encoder.{key}")
            for candidate in candidates:
                if candidate in target and target[candidate].shape == value.shape:
                    compatible[candidate] = value
                    break

        if not compatible:
            message = f"CaVE checkpoint has no compatible encoder parameters: {path}"
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, stacklevel=2)
            return
        missing, _ = self.cave.load_state_dict(compatible, strict=False)
        if strict and missing:
            raise RuntimeError(f"CaVE checkpoint is incomplete; missing {len(missing)} parameters")

    def _freeze_cave(self) -> None:
        self.cave.requires_grad_(False)
        self.cave.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_cave:
            self.cave.eval()
        return self

    def _init_trainable_weights(self) -> None:
        nn.init.normal_(self.prior_tokens, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)
        modules = [self.input_proj, self.query_attention, self.output_norm]
        for root in modules:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.config.cave_in_nc:
            raise ValueError("images must have shape (B, cave_in_nc, H, W)")
        if self.config.freeze_cave:
            with torch.no_grad():
                return self.cave.get_visual_embedding(images)
        return self.cave.get_visual_embedding(images)

    def _aggregate(self, visual_embedding: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, width = visual_embedding.shape
        if width != self.config.cave_embed_dim:
            raise ValueError("CaVE feature width does not match cave_embed_dim")
        if self.pos_embed is not None and sequence_length > self.pos_embed.shape[1]:
            raise ValueError("CaVE token sequence exceeds max_seq_len")

        context = self.input_proj(visual_embedding)
        if self.pos_embed is not None:
            context = context + self.pos_embed[:, :sequence_length]
        queries = self.prior_tokens.expand(batch, -1, -1)
        for attention in self.query_attention:
            queries = attention(queries, context)
        return self.output_norm(queries)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._aggregate(self._encode_images(images))

    def forward_dual_view(
        self,
        primary_images: torch.Tensor,
        wrist_images: torch.Tensor,
    ) -> torch.Tensor:
        if primary_images.shape[0] != wrist_images.shape[0]:
            raise ValueError("primary and wrist batches must have the same size")
        primary = self._encode_images(primary_images)
        wrist = self._encode_images(wrist_images)
        return self._aggregate(torch.cat([primary, wrist], dim=1))

    def get_cave_visual_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return self._encode_images(images)

    def get_projected_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return self.input_proj(self._encode_images(images))

    def get_decoder_output(self, images: torch.Tensor):
        """Return CaVE auxiliary outputs for post-training diagnostics."""
        if self.config.freeze_cave:
            with torch.no_grad():
                return self.cave(images)
        return self.cave(images)

    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def num_tokens(self) -> int:
        return self.config.num_prior_tokens

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.config.hidden_dim}, "
            f"num_prior_tokens={self.config.num_prior_tokens}, "
            f"num_attention_layers={self.config.num_attention_layers}"
        )


# The paper uses CPE; CPFE is retained for checkpoint and script compatibility.
CPE = CPFE
CompressionPriorExtractor = CPFE
