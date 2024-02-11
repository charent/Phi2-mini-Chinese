import pyarrow.parquet as pq
import pyarrow as pa
import ujson
import numpy as np
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from datasets import Dataset
import ujson
from unicodedata import normalize
from multiprocessing import Pool
import sys
import time

sys.path.extend(['.', '..'])
from utils.utils import DropDatasetDuplicate, NON_CHAR

def read_baike_json(buffer_siez: int=10240, max_len=512) -> list[str]:

    bd_baike_563w_file = './data/563w_baidubaike.json'
    baike_texts = []
    index = 0
    with open(bd_baike_563w_file, 'r', encoding='utf-8') as f:

        def process_none(s: str) -> str:
            if s: return s
            return ''
        
        while True:
            line = f.readline()
            if not line: break

            item = ujson.loads(line)
            cur_txt, cur_len = [], 0

            if not item['title']: continue

            temp_txt = f"{item['title']}：{process_none(item['summary'])}"
            
            cur_len += len(temp_txt)
            cur_txt.append(temp_txt)

            for section in item['sections']:

                # 太长的截断不要了
                if cur_len > max_len:
                    break
                
                title = f"{section['title']}：" if section['title'] else ""
                temp_txt = f"{title}{process_none(section['content'])}"
                
                cur_len += len(temp_txt)
                cur_txt.append(temp_txt)
            
            # normalize 处理\u3000 \xa0，全角转半角
            temp_txt =  normalize('NFKC', ''.join(cur_txt))

            if len(temp_txt) > max_len:
                # 从 max_len 开始找第一个句号，叹号
                # n, i = len(temp_txt), max_len
                # while i < n and temp_txt[i] not in ('。', '！'):
                #     i += 1
                i = len(temp_txt) - 1
                while i > 0 and temp_txt[i] not in ('。', '！'):
                    i -= 1
                if i <= 0: i = len(temp_txt)

                temp_txt = ''.join(temp_txt[0: i + 1])
            
            baike_texts.append( (index, temp_txt) )
            index += 1

            if len(baike_texts) >= buffer_siez:
                yield baike_texts
                baike_texts = []

        if not baike_texts:
            yield baike_texts

def init_lock(l):
    global lock
    lock = l

def init_data_lsh(dl):
    global data_lsh
    data_lsh = dl

def get_doc_mini_hash(args: tuple, n_gram: int=3) -> MinHash:
    '''
    获取一段文本的mini hash
    '''

    index, doc, num_perm = args
    # 删除符号
    doc = ''.join(NON_CHAR.split(doc))

    # n元组切分
    docs = [doc[i: i + n_gram] for i in range(0, len(doc))]

    mini_hash = MinHash(num_perm=num_perm)
    for s in docs:
        mini_hash.update(s.encode('utf-8'))
    return index, mini_hash


if __name__ == '__main__':

     # =========================多进程======================================
    start = time.time()
    data_lsh = MinHashLSH(threshold=0.90, num_perm=128)
    num_perm=128
    i = 0
    total_txts = []

    # 启动 num_process 个进程处理，请根据自己的机器性能调整
    num_process = 8
    with Pool(processes=num_process) as pool:
        
        for baike_texts in read_baike_json(buffer_siez=100000, max_len=512):
            print(f'processed docs: {len(total_txts)}')

            total_txts.extend(baike_texts)
            
            lines = [(i, doc, num_perm) for i, doc in baike_texts]

            result = pool.map(get_doc_mini_hash, lines, chunksize=16)
        
            for index, doc_hash in result:
                data_lsh.insert(index, doc_hash)
       
    duplicate_set = set()
    for index, doc_hash in result:
        if index not in duplicate_set:
            closes = data_lsh.query(doc_hash)
            duplicate_set |= set(closes) - {index}

    end = time.time()
    print(f'multi process cost time: {end - start} s')
   
    # 去重
    final_txts = [ doc for i, doc in total_txts if i not in duplicate_set]

    print(f'origin data size: {len(total_txts)}, total duplicate size: {len(duplicate_set)}, no duplicate data siez: {len(final_txts)}')

    tb = pa.Table.from_arrays([final_txts], names=['text'])
    file_name = f'./data/baike_no_duplicate.parquet'
    pq.write_table(table=tb, where=file_name, row_group_size=50000, )

    print(f"save to {file_name}")

    with open('./data/duplicate_index.json', 'w', encoding='utf-8') as f:
        lst = list(duplicate_set)
        lst.sort()
        ujson.dump(lst, f, ensure_ascii=False, indent=4)

    # =========================单进程======================================
    # data_lsh = MinHashLSH(threshold=0.90, num_perm=128)
    # start = time.time()

    # result = []
    # for i, doc in enumerate(baike_texts):
    #     index, doc_hash = get_doc_mini_hash((i, doc, num_perm))
    #     result.append((index, doc_hash))
    #     data_lsh.insert(index, doc_hash)

    # duplicate_set = set()
    # for index, doc_hash in result:
    #     if index not in duplicate_set:
    #         closes = data_lsh.query(doc_hash)
    #         duplicate_set |= set(closes) - {index}

    # end = time.time()
    # print(f'sigle process cost time: {end - start} s')
    # print(f'total duplicate size: {len(duplicate_set)}')

    # with open('./duplicate-1p.json', 'w', encoding='utf-8') as f:
    #     lst = list(duplicate_set)
    #     lst.sort()
    #     ujson.dump(lst, f, ensure_ascii=False, indent=4)
    

    

    