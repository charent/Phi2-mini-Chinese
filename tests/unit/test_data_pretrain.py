"""预训练数据映射：bos/eos 插入、分块长度、dtype 选择。"""

from __future__ import annotations

import numpy as np

from phi2_mini.data.common import select_map_dtype
from phi2_mini.data.pretrain import token_to_id


def test_merges_docs_and_adds_bos_eos(stub_tokenizer):
    out = token_to_id({'text': ['a', 'b']}, stub_tokenizer, np.uint16, max_len=5)
    # bos=1, eos=2：a -> [1,10,11,12,2]；b -> [1,20,21,2]；按 max_len=5 切块
    assert [len(item) for item in out['input_ids']] == [5, 4]
    assert out['input_ids'][0].tolist() == [1, 10, 11, 12, 2]
    assert out['input_ids'][1].tolist() == [1, 20, 21, 2]


def test_chunk_count_is_ceil(stub_tokenizer):
    out = token_to_id({'text': ['a', 'b']}, stub_tokenizer, np.uint16, max_len=4)
    # 总长度 9，max_len=4 -> [4, 4, 1]
    assert [len(item) for item in out['input_ids']] == [4, 4, 1]


def test_uses_requested_dtype(stub_tokenizer):
    out = token_to_id({'text': ['a']}, stub_tokenizer, np.uint16, max_len=16)
    assert out['input_ids'][0].dtype == np.uint16


def test_empty_batch_returns_empty_list(stub_tokenizer):
    assert token_to_id({'text': []}, stub_tokenizer, np.uint16, max_len=8) == {'input_ids': []}


def test_select_map_dtype_boundary():
    assert select_map_dtype(1) is np.uint16
    assert select_map_dtype(65534) is np.uint16
    assert select_map_dtype(65535) is np.uint32
    assert select_map_dtype(65536) is np.uint32
