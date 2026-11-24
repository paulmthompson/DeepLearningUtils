"""Tests for PyTorch 2D axial RoPE and QKV compression post-init warnings."""

from __future__ import annotations

import warnings

import pytest
import torch
from DeepLearningUtils.Layers.Attention.SequentialCoAttention import (
    sequential_co_attention_config as co_attn_cfg,
)
from DeepLearningUtils.Layers.RotaryPositionalEncoding.rotary2d_axial_pytorch import (
    AxialRotary2DConfig,
    RotaryPositionalEncoding2D,
    rotate_half_2d,
)


def test_axial_rope_config_validation_and_warnings() -> None:
    """AxialRotary2DConfig enforces divisibility by 4 and warns when head_dim < 32."""
    with pytest.raises(ValueError, match="divisible by 4"):
        AxialRotary2DConfig(head_dim=18)
    with pytest.raises(TypeError, match="head_dim must be an integer"):
        AxialRotary2DConfig(head_dim=True)  # type: ignore[arg-type]

    with pytest.warns(UserWarning, match=r"head_dim=16 \(4 rotary frequency pairs"):
        cfg_small = AxialRotary2DConfig(head_dim=16)
    assert cfg_small.freqs_per_axis == 4

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg_ok = AxialRotary2DConfig(head_dim=64, max_h=32, max_w=32)
    assert cfg_ok.freqs_per_axis == 16
    assert len(caught) == 0


def test_axial_rope_multi_frame_tiling_and_shift_equivariance() -> None:
    """2D axial RoPE preserves norms, tiles multi-frame keys, and encodes 2D relative shifts."""
    torch.manual_seed(7)
    rope = RotaryPositionalEncoding2D(head_dim=32, max_h=8, max_w=8)

    q_single = torch.randn(2, 4, 16, 32)
    k_two_frames = q_single.repeat(1, 1, 2, 1)
    q_rot, k_rot = rope(q=q_single, k=k_two_frames, q_h=4, q_w=4, k_h=4, k_w=4)
    assert q_rot is not None
    assert k_rot is not None
    torch.testing.assert_close(q_rot.norm(dim=-1), q_single.norm(dim=-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_rot[:, :, :16], k_rot[:, :, 16:], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        rotate_half_2d(rotate_half_2d(q_single)), -q_single, atol=1e-6, rtol=1e-6
    )


def test_co_memory_attention_config_compression_warnings() -> None:
    """CoMemoryAttentionConfig warns on >2x Q/K or aggressive V compression."""
    with pytest.warns(UserWarning, match="key_dim"):
        co_attn_cfg.CoMemoryAttentionConfig(
            query_shape=(1, 16, 16, 384),
            memory_shape=(1, 16, 16, 384),
            key_dim=128,
            value_dim=256,
            attention_heads=4,
        )

    with pytest.warns(UserWarning, match="value_dim"):
        co_attn_cfg.CoMemoryAttentionConfig(
            query_shape=(1, 16, 16, 384),
            memory_shape=(1, 16, 16, 384),
            key_dim=192,
            value_dim=128,
            attention_heads=4,
        )
