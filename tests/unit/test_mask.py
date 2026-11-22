"""doc 级 attention mask 的 golden test（重构前后的行为锚点）。"""

from __future__ import annotations

import numpy as np

from phi2_mini.data.mask import create_efficiency_mask

EOS = 1


def test_mask_matches_documented_example():
    input_ids = np.array([2, 3, 4, 5, 1, 2, 3, 1])
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(create_efficiency_mask(input_ids, EOS), expected)


def test_without_eos_is_plain_causal_mask():
    input_ids = np.array([5, 6, 7])
    np.testing.assert_array_equal(create_efficiency_mask(input_ids, EOS), np.tril(np.ones((3, 3), dtype=np.int64)))


def test_none_eos_is_plain_causal_mask():
    mask = create_efficiency_mask(np.array([1, 2, 3]), None)
    np.testing.assert_array_equal(mask, np.tril(np.ones((3, 3), dtype=np.int64)))


def test_single_token():
    np.testing.assert_array_equal(create_efficiency_mask(np.array([7]), EOS), np.array([[1]]))


def test_only_own_doc_is_visible():
    # 两个 doc：doc0 = [2,3,eos]，doc1 = [4,5,eos]
    mask = create_efficiency_mask(np.array([2, 3, EOS, 4, 5, EOS]), EOS)
    np.testing.assert_array_equal(mask[:3, 3:], np.zeros((3, 3), dtype=np.int64))
    np.testing.assert_array_equal(mask[3:, :3], np.zeros((3, 3), dtype=np.int64))


def test_empty_input():
    assert create_efficiency_mask(np.array([], dtype=np.int64), EOS).shape == (0, 0)
