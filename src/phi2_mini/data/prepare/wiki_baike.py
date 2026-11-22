"""清洗百度百科 / wikipedia 原始 json，按 MinHash 去重后写成 parquet。

用法（在仓库根目录执行）::

    uv run python -m phi2_mini.data.prepare.wiki_baike \\
        --source wiki \\
        --input data/wiki-zh/wikipedia-cn-20230720-filtered.json \\
        --output data/wiki_25w.parquet

    uv run python -m phi2_mini.data.prepare.wiki_baike \\
        --source baike \\
        --input data/563w_baidubaike.json \\
        --output data/baike_no_duplicate.parquet

重构前的两个问题在这里修掉：
1. 末尾 ``if not baike_texts: yield ...`` 条件写反，导致最后不满一批的数据被丢掉；
2. 去重只对「最后一个批次」做 LSH 查询（``result`` 变量泄漏到循环外），跨批次重复漏检。
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, TypeAlias
from unicodedata import normalize

import pyarrow as pa
import pyarrow.parquet as pq
import ujson

from phi2_mini.data.dedup import (
    DEFAULT_N_GRAM,
    DEFAULT_NUM_PERM,
    DEFAULT_THRESHOLD,
    get_doc_mini_hash,
)
from phi2_mini.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BUFFER_SIZE = 40_960
DEFAULT_MAX_LEN = 512

TextBatch: TypeAlias = list[tuple[int, str]]


def _safe_text(text: Optional[str]) -> str:
    return text or ''


def _truncate_at_punctuation(text: str, max_len: int) -> str:
    """超长文本从末尾回退到最近的句号/感叹号处截断。"""
    if len(text) <= max_len:
        return text

    index = len(text) - 1
    while index > 0 and text[index] not in ('。', '！'):
        index -= 1
    if index <= 0:
        index = len(text)
    return text[: index + 1]


def read_baike_json(
    file: str | Path,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    max_len: int = DEFAULT_MAX_LEN,
) -> Iterator[TextBatch]:
    """流式读取百度百科 jsonl，每条产出 ``[(index, text), ...]``。"""
    batch: TextBatch = []
    index = 0

    with open(file, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue

            item = json.loads(line)
            if not item['title']:
                continue

            parts: list[str] = [f"{item['title']}：{_safe_text(item['summary'])}"]
            current_len = len(parts[0])

            for section in item['sections']:
                if current_len > max_len:
                    break
                title = f"{section['title']}：" if section['title'] else ''
                text = f"{title}{_safe_text(section['content'])}"
                current_len += len(text)
                parts.append(text)

            # normalize 处理 \u3000 / \xa0，全角转半角
            text = normalize('NFKC', ''.join(parts))
            batch.append((index, _truncate_at_punctuation(text, max_len)))
            index += 1

            if len(batch) >= buffer_size:
                yield batch
                batch = []

    if batch:  # 修复：原来写成 `if not batch`，最后一批会被丢弃
        yield batch


def read_wiki(
    file: str | Path,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    max_len: int = DEFAULT_MAX_LEN,
) -> Iterator[TextBatch]:
    """读取 wikipedia-cn 的 json（整个文件是一个 list）。"""
    with open(file, encoding='utf-8') as handle:
        items = ujson.load(handle)

    batch: TextBatch = []
    index = 0
    for item in items:
        text = item['completion']
        if len(text) < 5:
            continue

        text = normalize('NFKC', text)
        batch.append((index, text))
        index += 1

        if len(batch) >= buffer_size:
            yield batch
            batch = []

    if batch:
        yield batch


def process_to_dataset(
    save_to_file: str | Path,
    read_txt_list_func: Callable[..., Iterator[TextBatch]],
    func_args: Optional[dict] = None,
    num_process: int = 8,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    n_gram: int = DEFAULT_N_GRAM,
) -> Path:
    """标准化 + MinHash 去重 + 落盘为 parquet。

    Args:
        save_to_file: 输出去重后的 parquet 路径。
        read_txt_list_func: 生成器函数，逐批产出 ``[(index, text), ...]``。
        func_args: 传给 ``read_txt_list_func`` 的参数。
        num_process: 计算 MinHash 的进程数，按机器性能调整。
        threshold / num_perm / n_gram: 去重参数。
    """
    from datasketch import MinHashLSH

    data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    total_texts: TextBatch = []
    duplicate_indices: set[int] = set()
    read_args = func_args or {}

    # multiprocessing.Pool 是工厂函数而非类型，这里用 Any 表示进程池
    def handle_batch(pool: Optional[Any], batch: TextBatch) -> None:
        total_texts.extend(batch)
        tasks = [(index, text, num_perm) for index, text in batch]
        if pool is None:
            hashes = [get_doc_mini_hash(task, n_gram=n_gram) for task in tasks]
        else:
            hashes = pool.map(partial(_hash_task, n_gram=n_gram), tasks, chunksize=16)

        # 逐条 query + insert：跨批次的重复也能被检出
        for index, doc_hash in hashes:
            if data_lsh.query(doc_hash):
                duplicate_indices.add(index)
            data_lsh.insert(index, doc_hash)

        logger.info('已处理 %d 条，累计重复 %d 条', len(total_texts), len(duplicate_indices))

    if num_process > 1:
        with Pool(processes=num_process) as pool:
            for batch in read_txt_list_func(**read_args):
                handle_batch(pool, batch)
    else:
        for batch in read_txt_list_func(**read_args):
            handle_batch(None, batch)

    final_texts = [text for index, text in total_texts if index not in duplicate_indices]
    logger.info('原始 %d 条，重复 %d 条，保留 %d 条', len(total_texts), len(duplicate_indices), len(final_texts))

    table = pa.Table.from_arrays([pa.array(final_texts, type=pa.string())], names=['text'])
    output = Path(save_to_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table=table, where=output, row_group_size=50_000)
    logger.info('已保存 %s', output)
    return output


def _hash_task(task: tuple[int, str, int], n_gram: int = DEFAULT_N_GRAM) -> tuple[int, object]:
    """``Pool.map`` 的顶层包装（默认参数保证可 pickle）。"""
    return get_doc_mini_hash(task, n_gram=n_gram)


READERS: dict[str, Callable[..., Iterator[TextBatch]]] = {
    'baike': read_baike_json,
    'wiki': read_wiki,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='清洗并去重百度百科 / wikipedia 语料')
    parser.add_argument('--source', choices=sorted(READERS), required=True, help='原始数据格式')
    parser.add_argument('--input', required=True, help='原始 json / jsonl 文件')
    parser.add_argument('--output', required=True, help='输出 parquet 路径')
    parser.add_argument('--num-process', type=int, default=8, help='MinHash 进程数')
    parser.add_argument('--buffer-size', type=int, default=DEFAULT_BUFFER_SIZE, help='每批文档数')
    parser.add_argument('--max-len', type=int, default=DEFAULT_MAX_LEN, help='单条文档最大字符数')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='相似度阈值')
    parser.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash 置换数')
    parser.add_argument('--n-gram', type=int, default=DEFAULT_N_GRAM, help='shingle 的 n-gram 大小')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    process_to_dataset(
        save_to_file=args.output,
        read_txt_list_func=READERS[args.source],
        func_args={'file': args.input, 'buffer_size': args.buffer_size, 'max_len': args.max_len},
        num_process=args.num_process,
        threshold=args.threshold,
        num_perm=args.num_perm,
        n_gram=args.n_gram,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
