"""MinHash 去重逻辑。"""

from __future__ import annotations

import pytest

pytest.importorskip('datasketch')

from phi2_mini.data.dedup import (
    DropDatasetDuplicate,
    build_min_hash,
    clean_text,
    get_dataset_duplicate_index,
    group_duplicates,
    shingles,
)


def test_clean_text_removes_punctuation():
    assert clean_text('你好，世界！123_abc') == '你好世界123_abc'


def test_shingles():
    assert shingles('abcdef', 3) == ['abc', 'bcd', 'cde', 'def']
    assert shingles('abc', 1) == ['a', 'b', 'c']
    assert shingles('ab', 3) == []


def test_punctuation_does_not_affect_hash():
    assert build_min_hash('你好，世界！', num_perm=16) == build_min_hash('你好世界', num_perm=16)


def test_identical_docs_keep_first():
    data = ['你好世界，欢迎来到这里', '你好世界，欢迎来到这里', '完全不同的一段文字内容啊']
    assert get_dataset_duplicate_index(data, num_perm=64) == {1}


def test_no_duplicates():
    data = ['第一段完全不同的文本内容', '第二段也不一样的文字材料', '第三段又是另一个句子']
    assert get_dataset_duplicate_index(data, num_perm=64) == set()


def test_streaming_deduplicator():
    deduper = DropDatasetDuplicate(num_perm=64)
    deduper.add_doc(0, '你好世界，欢迎来到这里')
    deduper.add_doc(1, '你好世界，欢迎来到这里')
    deduper.add_doc(2, '完全不同的一段文字内容啊')
    assert deduper.get_duplicate_indices() == {1}
    assert deduper.get_duplicate_indexs() == {1}


def test_group_duplicates():
    data = ['你好世界，欢迎来到这里', '你好世界，欢迎来到这里', '你好世界，欢迎来到这里']
    assert group_duplicates(data, num_perm=64) == {0: {1, 2}}
