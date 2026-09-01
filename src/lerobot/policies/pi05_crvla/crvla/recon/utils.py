"""Utilities for selecting visual tokens from VLM hidden states."""

from collections.abc import Sequence

import torch

from .config import ReconHeadConfig


def extract_vision_tokens(
    hidden_states: Sequence[torch.Tensor],
    layer_idx: int,
    num_patches: int,
    offset: int = 1,
) -> torch.Tensor:
    """Extract visual tokens from ``[BOS, vision, text/action]`` hidden states."""
    if not hidden_states:
        raise ValueError("hidden_states is empty; enable output_hidden_states")
    if not -len(hidden_states) <= layer_idx < len(hidden_states):
        raise IndexError("layer_idx is outside the hidden-state sequence")
    if num_patches <= 0 or offset < 0:
        raise ValueError("num_patches must be positive and offset non-negative")

    layer_hidden = hidden_states[layer_idx]
    if layer_hidden.ndim != 3:
        raise ValueError("each hidden state must have shape (B, sequence, D)")
    if offset + num_patches > layer_hidden.shape[1]:
        raise ValueError("the hidden-state sequence is too short for the requested visual tokens")
    return layer_hidden[:, offset : offset + num_patches, :]


def extract_vision_tokens_with_config(
    hidden_states: Sequence[torch.Tensor],
    num_patches: int,
    config: ReconHeadConfig,
) -> torch.Tensor:
    return extract_vision_tokens(
        hidden_states,
        layer_idx=config.layer_idx,
        num_patches=num_patches,
        offset=config.vision_token_offset,
    )


def count_recon_parameters(module: torch.nn.Module) -> dict:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}
