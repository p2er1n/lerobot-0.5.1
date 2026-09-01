#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch

from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.pi05_crvla.configuration_pi05_crvla import PI05CRVLAConfig
from lerobot.policies.pi05_crvla.modeling_pi05_crvla import PI05CRVLAPytorch
from lerobot.policies.pi05_crvla.crvla import (
    CPFE,
    ChannelParameterHead,
    ChannelParameterHeadConfig,
    CompressionPriorActionAnchorBlock,
    CPFEConfig,
    CPFEFusion,
    FeatureReconstructionLoss,
    ReconHeadConfig,
    ReconLossConfig,
    VLMFeatureReconstructionHead,
)


def test_pi05_crvla_config_registration():
    config = PI05CRVLAConfig(
        device="cpu",
        crvla_cave_channels=(4, 8, 12, 16),
        crvla_cave_num_blocks=1,
    )
    assert config.type == "pi05_crvla"
    assert isinstance(make_policy_config("pi05_crvla", device="cpu"), PI05CRVLAConfig)
    assert get_policy_class("pi05_crvla").config_class is PI05CRVLAConfig

    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    assert preprocessor is not None
    assert postprocessor is not None


def test_cpe_and_cam_shapes():
    cpe = CPFE(
        CPFEConfig(
            cave_nc=[4, 8, 12, 16],
            cave_nb=1,
            cave_embed_dim=16,
            freeze_cave=False,
            hidden_dim=16,
            num_prior_tokens=3,
            num_attention_layers=1,
            num_heads=4,
            dropout=0.0,
            max_seq_len=8,
        )
    )
    primary = torch.randn(2, 3, 32, 32)
    wrist = torch.randn(2, 3, 32, 32)
    priors = cpe.forward_dual_view(primary, wrist)
    assert priors.shape == (2, 3, 16)

    visual_tokens = torch.randn(2, 11, 16)
    cam = CPFEFusion(embed_dim=16, num_heads=4, num_layers=2, dropout=0.0)
    restored = cam(visual_tokens, priors)
    assert restored.shape == visual_tokens.shape
    assert cam.get_gate_value().item() < 0.02


def test_compression_prior_ignores_fully_masked_empty_camera():
    cpe = CPFE(
        CPFEConfig(
            cave_nc=[4, 8, 12, 16],
            cave_nb=1,
            cave_embed_dim=16,
            freeze_cave=False,
            hidden_dim=16,
            num_prior_tokens=3,
            num_attention_layers=1,
            num_heads=4,
            dropout=0.0,
            max_seq_len=8,
        )
    )
    policy_model = PI05CRVLAPytorch.__new__(PI05CRVLAPytorch)
    torch.nn.Module.__init__(policy_model)
    policy_model.compression_prior_extractor = cpe
    images = [torch.randn(2, 3, 32, 32) for _ in range(3)]
    masks = [
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    ]

    priors = policy_model._extract_compression_prior(images, masks)

    assert priors.shape == (2, 3, 16)


def test_action_anchor_is_identity_initialized_and_prior_conditioned():
    anchor = CompressionPriorActionAnchorBlock(
        action_dim=16,
        vlm_dim=24,
        prior_dim=12,
        num_heads=4,
        dropout=0.0,
    )
    actions = torch.randn(2, 5, 16, requires_grad=True)
    vlm = torch.randn(2, 7, 24)
    prior_a = torch.randn(2, 3, 12, requires_grad=True)
    prior_b = torch.randn(2, 3, 12)

    torch.testing.assert_close(anchor(actions, vlm, prior_a), actions)

    with torch.no_grad():
        anchor.output_projection.weight.normal_(std=0.02)
        anchor.ffn[-1].weight.normal_(std=0.02)
        anchor.alpha.fill_(0.5)
    output_a = anchor(actions, vlm, prior_a)
    output_b = anchor(actions, vlm, prior_b)
    assert not torch.allclose(output_a, output_b)

    output_a.square().mean().backward()
    assert prior_a.grad is not None
    assert prior_a.grad.abs().sum() > 0


def test_reconstruction_and_channel_training_modules():
    reconstruction_head = VLMFeatureReconstructionHead(
        ReconHeadConfig(
            embed_dim=16,
            bottleneck_dim=8,
            patch_grid_size=4,
            num_mixer_blocks=1,
            num_groups=4,
        )
    )
    compressed = torch.randn(2, 32, 16, requires_grad=True)
    clean = torch.randn(2, 32, 16, requires_grad=True)
    reconstructed, residual = reconstruction_head.forward_with_residual(compressed)
    torch.testing.assert_close(reconstructed, compressed)

    criterion = FeatureReconstructionLoss(ReconLossConfig(direct_alignment_weight=0.2, residual_weight=1e-3))
    reconstruction_loss, metrics = criterion(reconstructed, clean, compressed, residual)
    assert torch.isfinite(reconstruction_loss)
    assert "recon_direct_alignment" in metrics
    reconstruction_loss.backward()
    assert compressed.grad is not None
    assert clean.grad is None

    channel_head = ChannelParameterHead(
        ChannelParameterHeadConfig(
            input_dim=16,
            hidden_dim=8,
            num_codecs=4,
            num_continuous_parameters=2,
            dropout=0.0,
        )
    )
    channel_outputs = channel_head(torch.randn(2, 3, 16))
    assert channel_outputs["codec_logits"].shape == (2, 4)
    assert channel_outputs["continuous"].shape == (2, 2)
