"""基于 MinHash LSH 的文本去重。

重构前 ``utils/utils.py`` 里同时存在两种粒度：``get_doc_mini_hash`` 用 3-gram，
而 ``add_doc`` / ``get_dataset_duplicate_index`` 实际按单字符做 shingle，导致
两处「相似」语义不一致。这里统一为可配置的 n-gram（默认 3），阈值行为保持一致。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence, cast

from datasketch import MinHash, MinHashLSH

# 只保留中文、英文、下划线、数字，去掉标点
NON_CHAR = re.compile('[^[\u4E00-\u9FA5|A-Za-z_0-9]')

DEFAULT_N_GRAM = 3
DEFAULT_NUM_PERM = 256
DEFAULT_THRESHOLD = 0.85


def clean_text(doc: str) -> str:
    """去掉标点等无关字符。"""
    return ''.join(NON_CHAR.split(doc))


def shingles(doc: str, n_gram: int = DEFAULT_N_GRAM) -> list[str]:
    """把文本切成 n-gram 片段；``n_gram <= 1`` 时退化为单字符。"""
    if n_gram <= 1:
        return list(doc)
    return [doc[i : i + n_gram] for i in range(max(len(doc) - n_gram + 1, 0))]


def build_min_hash(doc: str, num_perm: int = DEFAULT_NUM_PERM, n_gram: int = DEFAULT_N_GRAM) -> MinHash:
    """计算一段文本的 MinHash 签名。"""
    mini_hash = MinHash(num_perm=num_perm)
    for shingle in shingles(clean_text(doc), n_gram):
        mini_hash.update(shingle.encode('utf-8'))
    return mini_hash


def _query(lsh: MinHashLSH, doc_hash: MinHash) -> list[int]:
    """``MinHashLSH.query`` 在类型桩里返回 ``set[Hashable]``，实际就是我们插入的 int 索引。"""
    return cast('list[int]', list(lsh.query(doc_hash)))


def get_doc_mini_hash(args: tuple[int, str, int], n_gram: int = DEFAULT_N_GRAM) -> tuple[int, MinHash]:
    """多进程 map 用的入口：``(index, doc, num_perm) -> (index, MinHash)``。"""
    index, doc, num_perm = args
    return index, build_min_hash(doc, num_perm=num_perm, n_gram=n_gram)


class DropDatasetDuplicate:
    """流式去重：逐条 ``add_doc``，最后用 ``get_duplicate_indices`` 取需要删除的索引。

    语义是「保留首次出现的文档」，即返回的索引都是后出现的重复项。
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        num_perm: int = DEFAULT_NUM_PERM,
        n_gram: int = DEFAULT_N_GRAM,
    ) -> None:
        self.num_perm = num_perm
        self.n_gram = n_gram
        self.data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.duplicate_indices: set[int] = set()

    def add_doc(self, index: int, doc: str) -> set[int]:
        """加入一个文档，返回与其相似的已存在文档索引。"""
        doc_hash = build_min_hash(doc, num_perm=self.num_perm, n_gram=self.n_gram)
        close_duplicates = set(_query(self.data_lsh, doc_hash))
        self.data_lsh.insert(index, doc_hash)
        if close_duplicates:
            self.duplicate_indices.add(index)
        return close_duplicates

    def add_doc_mutiprocess(self, args: tuple[int, str]) -> None:
        """多进程/旧接口兼容入口。"""
        index, doc = args
        self.add_doc(index, doc)

    def get_duplicate_indices(self) -> set[int]:
        """返回所有需要删除的（重复）文档索引。"""
        return set(self.duplicate_indices)

    # 兼容重构前的拼写
    def get_duplicate_indexs(self) -> set[int]:
        return self.get_duplicate_indices()


def get_dataset_duplicate_index(
    data: Iterable[str],
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    n_gram: int = DEFAULT_N_GRAM,
) -> set[int]:
    """返回数据集中需要删除的重复文档索引（保留首次出现）。

    Example:
        ``['a', 'b', 'a']`` -> ``{2}``
    """
    duplicate_indices: set[int] = set()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for index, doc in enumerate(data):
        doc_hash = build_min_hash(doc, num_perm=num_perm, n_gram=n_gram)
        if _query(lsh, doc_hash):
            duplicate_indices.add(index)
        lsh.insert(index, doc_hash)

    return duplicate_indices


def group_duplicates(data: Sequence[str], threshold: float = DEFAULT_THRESHOLD, num_perm: int = DEFAULT_NUM_PERM,
                     n_gram: int = DEFAULT_N_GRAM) -> dict[int, set[int]]:
    """按「首次出现的索引」聚合重复项，便于分析重复簇。"""
    similar_index_cluster: dict[int, set[int]] = defaultdict(set)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for index, doc in enumerate(data):
        doc_hash = build_min_hash(doc, num_perm=num_perm, n_gram=n_gram)
        close_duplicates = _query(lsh, doc_hash)
        lsh.insert(index, doc_hash)
        if close_duplicates:
            similar_index_cluster[min(close_duplicates)].add(index)

    return dict(similar_index_cluster)
