"""Configuration for VLM feature reconstruction."""

from dataclasses import dataclass


@dataclass
class ReconHeadConfig:
    """Configuration for the residual reconstruction head."""

    embed_dim: int = 896
    bottleneck_dim: int = 256
    patch_grid_size: int = 16
    num_mixer_blocks: int = 2
    conv_kernel_size: int = 3
    num_groups: int = 8
    dropout: float = 0.0
    layer_idx: int = 8
    vision_token_offset: int = 1

    def __post_init__(self) -> None:
        if self.embed_dim <= 0 or self.bottleneck_dim <= 0:
            raise ValueError("reconstruction dimensions must be positive")
        if self.patch_grid_size <= 0 or self.num_mixer_blocks <= 0:
            raise ValueError("grid and mixer-block counts must be positive")
        if self.conv_kernel_size <= 0 or self.conv_kernel_size % 2 != 1:
            raise ValueError("conv_kernel_size must be a positive odd integer")
        if self.num_groups <= 0 or self.bottleneck_dim % self.num_groups != 0:
            raise ValueError("bottleneck_dim must be divisible by num_groups")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.vision_token_offset < 0:
            raise ValueError("vision_token_offset must be non-negative")

    @property
    def num_patches_per_view(self) -> int:
        return self.patch_grid_size**2


@dataclass
class ReconLossConfig:
    """Configuration for Eq. (8)-(9) of the reconstruction objective."""

    smooth_l1_weight: float = 0.1
    smooth_l1_beta: float = 1.0
    direct_alignment_weight: float = 0.1
    residual_weight: float = 1e-4
    compute_in_fp32: bool = True
    max_loss_weight: float = 0.05
    curriculum_start_step: int = 0
    curriculum_end_step: int = 2000

    def __post_init__(self) -> None:
        weights = (
            self.smooth_l1_weight,
            self.direct_alignment_weight,
            self.residual_weight,
            self.max_loss_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("reconstruction loss weights must be non-negative")
        if self.smooth_l1_beta < 0:
            raise ValueError("smooth_l1_beta must be non-negative")
        if self.curriculum_start_step < 0:
            raise ValueError("curriculum_start_step must be non-negative")
        if self.curriculum_end_step < self.curriculum_start_step:
            raise ValueError("curriculum_end_step must be >= curriculum_start_step")
