"""按文本相似度去重 parquet 数据集（MinHash LSH），把结果写到新文件。

输入文件只需要一列 ``text``，输出同样只有一列 ``text``。

用法（在仓库根目录执行）::

    uv run python -m phi2_mini.data.prepare.drop_duplicate data/train.parquet data/train_no_duplicate.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from phi2_mini.data.dedup import (
    DEFAULT_N_GRAM,
    DEFAULT_NUM_PERM,
    DEFAULT_THRESHOLD,
    get_dataset_duplicate_index,
)
from phi2_mini.logging_utils import get_logger

logger = get_logger(__name__)


def drop_duplicate(
    src_file: str | Path,
    dst_file: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    n_gram: int = DEFAULT_N_GRAM,
) -> Path:
    """读取 ``src_file`` 的 text 列，去掉重复后写入 ``dst_file``，返回输出路径。"""
    start = time.time()
    texts = pq.read_table(src_file, columns=['text']).column('text').to_pylist()
    logger.info('原始数据量：%d', len(texts))

    duplicate_indices = get_dataset_duplicate_index(
        texts,
        threshold=threshold,
        num_perm=num_perm,
        n_gram=n_gram,
    )
    final_texts = [text for index, text in enumerate(texts) if index not in duplicate_indices]
    logger.info('重复 %d 条，保留 %d 条', len(duplicate_indices), len(final_texts))

    # 用 pa.array 显式指定类型：空结果时 from_arrays([[]]) 会推断不出 schema
    table = pa.Table.from_arrays([pa.array(final_texts, type=pa.string())], names=['text'])
    output = Path(dst_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table=table, where=output, row_group_size=50_000)
    logger.info('已保存 %s（耗时 %.1fs）', output, time.time() - start)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='按文本相似度去重 parquet 数据集')
    parser.add_argument('src_file', nargs='?', default='data/train.parquet', help='输入 parquet（一列 text）')
    parser.add_argument('dst_file', nargs='?', default='data/train_no_duplicate.parquet', help='输出去重后的 parquet')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='相似度阈值')
    parser.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash 置换数')
    parser.add_argument('--n-gram', type=int, default=DEFAULT_N_GRAM, help='shingle 的 n-gram 大小')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    drop_duplicate(
        args.src_file,
        args.dst_file,
        threshold=args.threshold,
        num_perm=args.num_perm,
        n_gram=args.n_gram,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
