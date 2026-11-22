"""SFT 数据映射：模板拼接顺序与对齐校验。"""

from __future__ import annotations

import numpy as np
import pytest

from phi2_mini.data.sft import INSTRUCTION_TEMPLATE, RESPONSE_TEMPLATE, token_to_id


def _template_ids(stub_tokenizer):
    return stub_tokenizer.encode(INSTRUCTION_TEMPLATE), stub_tokenizer.encode(RESPONSE_TEMPLATE)


def test_sample_layout(stub_tokenizer):
    out = token_to_id(
        {'instruction': ['a'], 'output': ['b']},
        stub_tokenizer,
        np.uint16,
        _template_ids(stub_tokenizer),
    )
    # [bos] + 提问模板 + 提问 + 回答模板 + 回答 + [eos]
    assert out['input_ids'][0].tolist() == [1, 30, 10, 11, 12, 31, 20, 21, 2]


def test_multiple_samples(stub_tokenizer):
    out = token_to_id(
        {'instruction': ['a', 'b'], 'output': ['b', 'a']},
        stub_tokenizer,
        np.uint16,
        _template_ids(stub_tokenizer),
    )
    assert len(out['input_ids']) == 2
    assert out['input_ids'][1].tolist() == [1, 30, 20, 21, 31, 10, 11, 12, 2]


def test_mismatched_columns_raise(stub_tokenizer):
    with pytest.raises(ValueError):
        token_to_id(
            {'instruction': ['a', 'b'], 'output': ['b']},
            stub_tokenizer,
            np.uint16,
            _template_ids(stub_tokenizer),
        )
