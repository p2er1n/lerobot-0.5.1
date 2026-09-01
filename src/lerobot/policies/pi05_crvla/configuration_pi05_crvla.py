#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Copyright 2026 Anonymous CR-VLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config


@PreTrainedConfig.register_subclass("pi05_crvla")
@dataclass
class PI05CRVLAConfig(PI05Config):
    """PI0.5 with CR-VLA compression-aware visual and action conditioning."""

    # Compression Prior Extractor (CPE / CPFE).
    crvla_cave_checkpoint: str | None = None
    crvla_cave_channels: tuple[int, int, int, int] = (128, 256, 512, 1024)
    crvla_cave_num_blocks: int = 4
    crvla_freeze_cave: bool = True
    crvla_strict_cave_checkpoint: bool = False
    crvla_num_prior_tokens: int = 8
    crvla_prior_attention_layers: int = 2
    crvla_attention_heads: int = 8
    crvla_dropout: float = 0.1
    crvla_max_prior_sequence_length: int = 1024

    # Compression-Aware Modulation (CAM).
    crvla_cam_layers: int = 2
    crvla_cam_mlp_ratio: float = 4.0
    crvla_cam_residual_gate_init: float = -4.0

    # Compression-prior Action Anchor for the PI0.5 flow-matching expert.
    crvla_action_anchor_layers: int = 1
    crvla_action_anchor_ffn_ratio: float = 4.0

    # Optional training-only clean/compressed feature reconstruction objective.
    # Clean images are read from ``<image feature key><crvla_clean_image_suffix>``.
    crvla_reconstruction_enabled: bool = False
    crvla_clean_image_suffix: str = ".clean"
    crvla_reconstruction_bottleneck_dim: int = 256
    crvla_reconstruction_mixer_blocks: int = 2
    crvla_reconstruction_groups: int = 8
    crvla_reconstruction_smooth_l1_weight: float = 0.1
    crvla_reconstruction_direct_alignment_weight: float = 0.1
    crvla_reconstruction_residual_weight: float = 1e-4
    crvla_reconstruction_max_loss_weight: float = 0.05
    crvla_reconstruction_curriculum_start_step: int = 0
    crvla_reconstruction_curriculum_end_step: int = 2000

    # Optional removable channel-parameter prediction head.
    crvla_channel_head_enabled: bool = False
    crvla_channel_head_hidden_dim: int = 512
    crvla_num_codecs: int = 4
    crvla_num_channel_parameters: int = 2
    crvla_codec_target_key: str = "crvla.codec_id"
    crvla_channel_target_key: str = "crvla.channel_parameters"
    crvla_channel_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.crvla_cave_channels) != 4 or any(width <= 0 for width in self.crvla_cave_channels):
            raise ValueError("crvla_cave_channels must contain four positive widths")
        if self.crvla_cave_num_blocks <= 0:
            raise ValueError("crvla_cave_num_blocks must be positive")
        if self.crvla_num_prior_tokens <= 0 or self.crvla_prior_attention_layers <= 0:
            raise ValueError("CR-VLA prior token and attention-layer counts must be positive")
        if self.crvla_attention_heads <= 0:
            raise ValueError("crvla_attention_heads must be positive")
        model_widths = {"gemma_300m": 1024, "gemma_2b": 2048}
        if model_widths[self.paligemma_variant] % self.crvla_attention_heads != 0:
            raise ValueError("PaliGemma width must be divisible by crvla_attention_heads")
        if model_widths[self.action_expert_variant] % self.crvla_attention_heads != 0:
            raise ValueError("action expert width must be divisible by crvla_attention_heads")
        if not 0.0 <= self.crvla_dropout < 1.0:
            raise ValueError("crvla_dropout must be in [0, 1)")
        if self.crvla_max_prior_sequence_length <= 0:
            raise ValueError("crvla_max_prior_sequence_length must be positive")
        if self.crvla_cam_layers <= 0 or self.crvla_cam_mlp_ratio <= 0:
            raise ValueError("CR-VLA CAM layer count and MLP ratio must be positive")
        if self.crvla_action_anchor_layers < 0 or self.crvla_action_anchor_ffn_ratio <= 0:
            raise ValueError("CR-VLA Action Anchor settings are invalid")
        if not self.crvla_clean_image_suffix:
            raise ValueError("crvla_clean_image_suffix cannot be empty")
        if self.crvla_reconstruction_bottleneck_dim <= 0 or self.crvla_reconstruction_groups <= 0:
            raise ValueError("reconstruction bottleneck width and group count must be positive")
        if self.crvla_reconstruction_bottleneck_dim % self.crvla_reconstruction_groups != 0:
            raise ValueError("reconstruction bottleneck width must be divisible by its group count")
        if self.crvla_reconstruction_enabled and (
            self.image_resolution[0] != self.image_resolution[1] or self.image_resolution[0] % 14 != 0
        ):
            raise ValueError("CR-VLA reconstruction requires a square image resolution divisible by 14")
        if self.crvla_reconstruction_curriculum_end_step < self.crvla_reconstruction_curriculum_start_step:
            raise ValueError("reconstruction curriculum end must not precede its start")
        reconstruction_weights = (
            self.crvla_reconstruction_smooth_l1_weight,
            self.crvla_reconstruction_direct_alignment_weight,
            self.crvla_reconstruction_residual_weight,
            self.crvla_reconstruction_max_loss_weight,
        )
        if any(weight < 0 for weight in reconstruction_weights):
            raise ValueError("CR-VLA reconstruction weights cannot be negative")
        if self.crvla_num_codecs < 0 or self.crvla_num_channel_parameters < 0:
            raise ValueError("channel prediction output dimensions cannot be negative")
        if (
            self.crvla_channel_head_enabled
            and self.crvla_num_codecs == self.crvla_num_channel_parameters == 0
        ):
            raise ValueError("the CR-VLA channel head must predict at least one target")
        if self.crvla_channel_loss_weight < 0:
            raise ValueError("crvla_channel_loss_weight cannot be negative")


__all__ = ["DEFAULT_IMAGE_SIZE", "PI05CRVLAConfig"]
