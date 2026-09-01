"""Training-only VLM feature reconstruction components."""

from .config import ReconHeadConfig, ReconLossConfig
from .head import SpatialMixerBlock, VLMFeatureReconstructionHead
from .loss import (
    FeatureReconstructionLoss,
    curriculum_coefficient,
    feature_discrepancy,
    reconstruction_loss_weight,
)
from .utils import count_recon_parameters, extract_vision_tokens, extract_vision_tokens_with_config

__all__ = [
    "ReconHeadConfig",
    "ReconLossConfig",
    "VLMFeatureReconstructionHead",
    "SpatialMixerBlock",
    "FeatureReconstructionLoss",
    "feature_discrepancy",
    "curriculum_coefficient",
    "reconstruction_loss_weight",
    "extract_vision_tokens",
    "extract_vision_tokens_with_config",
    "count_recon_parameters",
]
