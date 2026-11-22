'''
按文本相似度去重 parquet 数据集（MinHash LSH），把去重后的结果写到新文件。

用法（在项目根目录执行）：
    uv run python dataset_prepare/drop_duplicate.py
    uv run python dataset_prepare/drop_duplicate.py ./data/train.parquet ./data/train_no_duplicate.parquet

输入文件只需要一列 `text`，输出文件同样只有一列 `text`。
'''
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.extend(['.', '..'])
from utils.utils import get_dataset_duplicate_index


def drop_duplicate(src_file: str, dst_file: str, threshold: float = 0.85, num_perm: int = 256) -> None:
    start = time.time()
    texts = pq.read_table(src_file, columns=['text']).column('text').to_pylist()
    print(f'origin data size: {len(texts)}')

    duplicate_idxs = get_dataset_duplicate_index(texts, threshold=threshold, num_perm=num_perm)
    final_texts = [text for i, text in enumerate(texts) if i not in duplicate_idxs]

    print(f'duplicate size: {len(duplicate_idxs)}, no duplicate size: {len(final_texts)}')

    table = pa.Table.from_arrays([final_texts], names=['text'])
    pq.write_table(table=table, where=dst_file, row_group_size=50000)
    print(f'save to {dst_file}, cost {time.time() - start:.1f} s')


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else './data/train.parquet'
    dst = sys.argv[2] if len(sys.argv) > 2 else './data/train_no_duplicate.parquet'
    drop_duplicate(src_file=src, dst_file=dst)
