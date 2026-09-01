"""Feature discrepancy and the complete reconstruction objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .config import ReconLossConfig


def feature_discrepancy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth_l1_weight: float,
    smooth_l1_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cosine distance plus weighted Smooth L1, averaged over batch and tokens."""
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    smooth_l1 = F.smooth_l1_loss(prediction, target, beta=smooth_l1_beta)
    return cosine + smooth_l1_weight * smooth_l1, cosine, smooth_l1


class FeatureReconstructionLoss(nn.Module):
    """Implement Eq. (9) with stop-gradient clean targets.

    ``compressed`` enables the direct alignment term. ``residual`` can be
    supplied explicitly or is inferred as ``reconstructed - compressed``.
    The two-argument call remains valid for reconstruction-only supervision.
    """

    def __init__(self, config: ReconLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        compressed: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if reconstructed.shape != target.shape:
            raise ValueError("reconstructed and target features must have identical shapes")
        if compressed is not None and compressed.shape != target.shape:
            raise ValueError("compressed and target features must have identical shapes")
        if residual is not None and residual.shape != target.shape:
            raise ValueError("residual and target features must have identical shapes")

        target = target.detach()
        if residual is None and compressed is not None:
            residual = reconstructed - compressed
        if self.config.compute_in_fp32:
            reconstructed = reconstructed.float()
            target = target.float()
            compressed = compressed.float() if compressed is not None else None
            residual = residual.float() if residual is not None else None

        reconstruction, cosine, smooth_l1 = feature_discrepancy(
            reconstructed,
            target,
            self.config.smooth_l1_weight,
            self.config.smooth_l1_beta,
        )
        direct = reconstruction.new_zeros(())
        if compressed is not None:
            direct, _, _ = feature_discrepancy(
                compressed,
                target,
                self.config.smooth_l1_weight,
                self.config.smooth_l1_beta,
            )
        residual_penalty = reconstruction.new_zeros(())
        if residual is not None:
            residual_penalty = residual.square().sum(dim=-1).mean()

        loss = (
            reconstruction
            + self.config.direct_alignment_weight * direct
            + self.config.residual_weight * residual_penalty
        )
        metrics = {
            "recon_loss": loss.detach().item(),
            "recon_discrepancy": reconstruction.detach().item(),
            "recon_cosine_loss": cosine.detach().item(),
            "recon_smooth_l1_loss": smooth_l1.detach().item(),
            "recon_direct_alignment": direct.detach().item(),
            "recon_residual_penalty": residual_penalty.detach().item(),
        }
        return loss, metrics


def curriculum_coefficient(step: int, config: ReconLossConfig) -> float:
    """Linearly increase the auxiliary objective coefficient from zero to one."""
    start, end = config.curriculum_start_step, config.curriculum_end_step
    if step <= start:
        return 0.0
    if step >= end or end == start:
        return 1.0
    return (step - start) / (end - start)


def reconstruction_loss_weight(step: int, config: ReconLossConfig) -> float:
    return config.max_loss_weight * curriculum_coefficient(step, config)
