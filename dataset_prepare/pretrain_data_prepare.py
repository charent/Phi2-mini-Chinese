from typing import Callable
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
from utils.utils import get_doc_mini_hash

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

def read_wiki(buffer_siez: int=10240, max_len=512) -> list[str]:
    '''
    数据来源: https://huggingface.co/datasets/bigscience-data/roots_zh-cn_wikipedia
    https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered
    '''

    wiki_file = './data/wiki-zh/wikipedia-cn-20230720-filtered.json'

    buffer_texts = []
    index = 0
    with open(wiki_file, 'r', encoding='utf-8') as f:
        lines = ujson.load(f)

        for item in lines:
 
            temp_txt = item['completion']

            if len(temp_txt) < 5: 
                continue
            
            # normalize 处理\u3000 \xa0，全角转半角
            temp_txt =  normalize('NFKC', ''.join(temp_txt))

            # if len(temp_txt) > max_len:
            #     # 从 max_len 开始找第一个句号，叹号
            #     n, i = len(temp_txt), max_len
            #     while i < n and temp_txt[i] not in ('。', '！'):
            #         i += 1
            
            #     temp_txt = ''.join(temp_txt[0: i + 1])
            
            buffer_texts.append( (index, temp_txt) )
            index += 1

            if len(buffer_texts) >= buffer_siez:
                yield buffer_texts
                buffer_texts = []

        if not buffer_texts:
            yield buffer_texts

def process_to_dataset(save_to_file: str, read_txt_list_func: Callable,  func_args: dict={}, num_process: int=8, ) -> None:
    '''
    将数据标准化、并利用mini hahs去重，最后保存到一个文件中
    save_to_file：去重后的文档保存到哪个文件
    read_txt_list_func：是一个读取原始数据集的函数，返回：list[tuple[int, str]]，列表中的每一个元素都是一个文档tuple，tuple的第一个元素为文档的索引，第二个元素为文档本身，在文档之间进行去重
    func_args：read_txt_list_func函数所需的参数
    num_process：进程个数，请根据自己机器的性能调整
    '''
    
    start = time.time()
    data_lsh = MinHashLSH(threshold=0.90, num_perm=128)
    num_perm=128
    total_txts = []

    with Pool(processes=num_process) as pool:
        
        for texts in read_txt_list_func(**func_args):
            print(f'processed docs: {len(total_txts)}')

            total_txts.extend(texts)
            
            lines = [(i, doc, num_perm) for i, doc in texts]

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

    pq.write_table(table=tb, where=save_to_file, row_group_size=50000, )

    print(f"save to {save_to_file}")

    # with open(f'./data/duplicate_index_{time.time()}.json', 'w', encoding='utf-8') as f:
    #     lst = list(duplicate_set)
    #     lst.sort()
    #     ujson.dump(lst, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':

    read_data_args = {
        'buffer_siez': 4_0960, 
        'max_len': 512,
    }

    # 处理百科数据
    # process_to_dataset(save_to_file='./data/baike_no_duplicate.parquet', read_txt_list_func=read_baike_json, func_args=read_data_args, num_process=8)

    # 处理wiki数据
    process_to_dataset(save_to_file='./data/wiki_25w.parquet', read_txt_list_func=read_wiki, func_args=read_data_args, num_process=8)
   

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

