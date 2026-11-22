"""attention 后端选择：缺 flash-attn 时必须回落，而不是硬传 flash_attention_2。"""

from __future__ import annotations

from phi2_mini.modeling import attention
from phi2_mini.modeling.attention import EAGER, FLASH_ATTENTION_2, resolve_attn_implementation


def test_uses_flash_attention_when_available():
    assert resolve_attn_implementation(True) == FLASH_ATTENTION_2


def test_falls_back_to_eager_when_unavailable():
    assert resolve_attn_implementation(False) == EAGER


def test_autodetect_without_flash_attn_returns_eager(monkeypatch):
    monkeypatch.setattr(attention, 'flash_attn_installed', lambda: False)
    assert resolve_attn_implementation() == EAGER


def test_autodetect_with_flash_attn_returns_flash(monkeypatch):
    monkeypatch.setattr(attention, 'flash_attn_installed', lambda: True)
    assert resolve_attn_implementation() == FLASH_ATTENTION_2
