"""模型配置构造：词表大小取整规则。"""

from __future__ import annotations

import pytest

from phi2_mini.modeling.factory import round_vocab_size


@pytest.mark.parametrize(
    ('vocab_size', 'expected'),
    [(0, 0), (1, 64), (63, 64), (64, 64), (65, 128), (64000, 64000), (64001, 64064)],
)
def test_round_vocab_size(vocab_size, expected):
    assert round_vocab_size(vocab_size) == expected
    assert expected % 64 == 0


def test_custom_multiple():
    assert round_vocab_size(100, multiple=32) == 128


def test_invalid_multiple():
    with pytest.raises(ValueError, match='multiple'):
        round_vocab_size(100, multiple=0)
