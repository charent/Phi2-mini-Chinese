"""Phi 4D mask 透传补丁：4D 转加性 mask、2D 走原生实现、patch 幂等。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from phi2_mini.modeling.packing import patch_phi_attention_mask  # noqa: E402


def test_patch_is_idempotent():
    patch_phi_attention_mask()
    assert patch_phi_attention_mask() is False


def test_4d_binary_mask_becomes_additive():
    from transformers.models.phi import modeling_phi

    patch_phi_attention_mask()

    binary = torch.tensor([[[[1, 0], [1, 1]]]])
    embeds = torch.zeros(1, 2, 4)

    mask = modeling_phi._prepare_4d_causal_attention_mask(binary, (1, 2), embeds, 0)

    assert mask is not None
    assert mask.shape == (1, 1, 2, 2)
    assert mask[0, 0, 0, 0].item() == 0
    assert mask[0, 0, 0, 1].item() < -1e30
    assert mask[0, 0, 1, 0].item() == 0


def test_2d_mask_still_uses_original_converter():
    from transformers.models.phi import modeling_phi

    patch_phi_attention_mask()

    mask_2d = torch.ones(1, 3)
    mask = modeling_phi._prepare_4d_causal_attention_mask(mask_2d, (1, 3), torch.zeros(1, 3, 4), 0)

    # 原生实现会返回 [bsz, 1, q_len, kv_len] 的加性 mask
    assert mask is not None
    assert mask.shape == (1, 1, 3, 3)
