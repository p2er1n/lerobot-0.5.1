"""Configuration objects for compression-prior modules."""

from dataclasses import dataclass, field


@dataclass
class CPFEConfig:
    """Configuration for the compression prior extractor (CPE)."""

    cave_checkpoint: str | None = None
    cave_in_nc: int = 3
    cave_out_nc: int = 3
    cave_nc: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    cave_nb: int = 4
    cave_act_mode: str = "BR"
    cave_embed_dim: int = 1024
    freeze_cave: bool = True
    strict_cave_checkpoint: bool = False

    hidden_dim: int = 2176
    num_prior_tokens: int = 32
    num_attention_layers: int = 1
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 1024
    use_pos_embed: bool = True

    def __post_init__(self) -> None:
        if not self.cave_nc:
            raise ValueError("cave_nc must contain at least one channel width")
        if self.cave_embed_dim != self.cave_nc[-1]:
            raise ValueError("cave_embed_dim must match the final cave_nc value")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.num_prior_tokens <= 0 or self.num_attention_layers <= 0:
            raise ValueError("token and attention-layer counts must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")


@dataclass
class CPFEFusionConfig:
    """Configuration for compression-aware visual modulation (CAM)."""

    embed_dim: int = 768
    num_heads: int = 8
    num_fusion_layers: int = 1
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    residual: bool = True
    residual_gate_init: float = -4.0

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.num_fusion_layers <= 0:
            raise ValueError("num_fusion_layers must be positive")


@dataclass
class ChannelParameterHeadConfig:
    """Configuration for the removable channel expert head."""

    input_dim: int = 2176
    hidden_dim: int = 512
    num_codecs: int = 4
    num_continuous_parameters: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("channel-head dimensions must be positive")
        if self.num_codecs < 0 or self.num_continuous_parameters < 0:
            raise ValueError("channel-head output dimensions must be non-negative")
        if self.num_codecs == 0 and self.num_continuous_parameters == 0:
            raise ValueError("the channel head must predict at least one target")
