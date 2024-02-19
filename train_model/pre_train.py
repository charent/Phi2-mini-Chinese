import os, platform, time, sys
from typing import Optional

import torch
from torch.utils.data import Sampler, SequentialSampler

from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling,Trainer, TrainingArguments, PhiForCausalLM, PhiConfig
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np

from utils.utils import MyTrainerCallback
from config import MyTrainArugment

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

attn_implementation = 'flash_attention_2'
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_implementation = 'eager' if platform.system() == 'Windows' else attn_implementation


def token_to_id(samples: dict[str, list], tokenizer: PreTrainedTokenizerFast, map_dtype: np.dtype, max_len: int) -> dict:

    batch_txt = samples['text']
    outputs = tokenizer(
        batch_txt,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    total_ids = []

    # 在每个文档的开始、结束添加bos、 eos
    for item in outputs["input_ids"]:
        total_ids += [bos_id] + item + [eos_id]
    
    n = len(total_ids) # 当前批次的总长度

    # 将一个批次的数据合并在一起，剩余的部分保留，如max_len=4, n=18, 则input_ids中数组的长度为[4, 4 ,4 , 4, 2]
    # 这样做的目的是减少pad_id在一个批次中的占比，提高GPU利用率
    input_ids = [np.array(total_ids[i: i + max_len], dtype=map_dtype) for i in range(0, n, max_len)]

    return {
            "input_ids": input_ids
        }

def get_maped_dataset(files: str|list[str], map_fun_args: dict) -> Dataset:
    '''
    获取数据集，请根据自己机器的性能更改 num_proc 的数量，以加快处理速度
    '''
    dataset = load_dataset(path='parquet', data_files=files, split='train', cache_dir='.cache')
    print(f'pretrain dataset doc size: {len(dataset)}, it will auto merge for efficiently training, the size will smaller then your provided.')

    maped_dataset = dataset.map(token_to_id, batched=True, batch_size=2_0000, fn_kwargs=map_fun_args, remove_columns=dataset.column_names, num_proc=8)
    return maped_dataset


class PreTrainTrainer(Trainer):

    # 预训练不需要打乱顺序
    def _get_train_sampler(self) -> Optional[Sampler]:
        return SequentialSampler(self.train_dataset)

def create_efficiency_mask(input_ids: np.array, eos_token_id: int) -> np.array:
    '''
    input_ids: np.array[int]
    return: np.array[np.array[int]]

    将多个短文本doc合并为超长文本后，需要mask其他不属于自己的doc。

    根据eos token_id所在的位置屏蔽左下三角矩阵（置零）,只关注自己的doc，屏蔽其他doc
    如:
    input_ids = [2,3,4,5,1,2,3,1], 1为eos id，其他为正常token id

      [[1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 1, 1, 1]]

    '''
    n = len(input_ids)
    idxs = np.where(input_ids == eos_token_id)[0]
    tri_mask = np.tril([[1] * n for _ in range(n)])
    
    start = 0
    for idx in idxs:
        tri_mask[idx + 1: , start: idx + 1] = 0
        start = idx
    
    return tri_mask

class MyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __call__(self, features, return_tensors=None):
        
        # 创建mask，同一行可能合并的多个doc，只关注自己所在doc，屏蔽其他doc
        eos_token_id = self.tokenizer.eos_token_id
        for item in features:
            item['attention_mask'] = create_efficiency_mask(np.array(item['input_ids']), eos_token_id)
      
        ret = super().__call__(features, return_tensors)
        
        attention_mask = ret['attention_mask'] # shape: [batch_size, q_len, kv_len], decoder-only模型，self-attention, q_len = kv_len = max_len
        
        ret['attention_mask'] = torch.unsqueeze(attention_mask, dim = 1) # shape: [batch_size, 1, q_len, kv_len]

        return  ret

def pre_train(config: MyTrainArugment):

    # # 0. 加载训练好的tokenizer
    # 如果你使用的`add_tokens`方法添加了自己的token，必须要用`len(tokenizer)`获取长度，`tokenizer.vocab_size`统计不包含你添加的字符。

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)

    # # 1. 定义模型
    # 从`config`定义，不是`from_pretrained`。 
    # 为了方便cuda计算，词表的大小注意一下，如果不是64的整数倍，可以手动向上取整为64的整数倍，也可以是其他 $2^x$ 数值的整数倍，如32、128、256都行。

    vocab_size = len(tokenizer)
    if vocab_size % 64 != 0:
        vocab_size = (vocab_size // 64 + 1) * 64
    print(f"source vocab size: {len(tokenizer)}, final vocab sieze: {vocab_size}")
    
    # ## token to id缓存到文件，使用的时候不用再次tokenize
    # 如果词表大小小于 65535 用uint16存储，节省磁盘空间，否则用uint32存储
    map_dtype = np.uint16 if vocab_size < 65535 else np.uint32
    
    map_fun_args = {
        'tokenizer': tokenizer,
        'map_dtype': map_dtype,
        'max_len': config.max_len,
    }

    train_dataset = get_maped_dataset(config.train_files, map_fun_args)
    if config.eval_file is not None:
        eval_dataset = get_maped_dataset(config.eval_file, map_fun_args)
    else:
        eval_dataset = None

    print(train_dataset, eval_dataset)

    # # 2. 定义data_collator
    # `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型
    data_collator = MyDataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 如果配置了flash_attention_2，请手动设置set_default_dtype为float16
    #  Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes.
    if attn_implementation == 'flash_attention_2':
        torch.set_default_dtype(torch.bfloat16)


    phi_config = PhiConfig(
        vocab_size=vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        max_position_embeddings=768,
        intermediate_size=4096,
        attn_implementation=attn_implementation,
        tie_word_embeddings=False,
    )

    model = PhiForCausalLM(phi_config)
    # model = model.to_bettertransformer()

    # 另外一个使用flash_attention_2的方法
    # model = PhiForCausalLM.from_pretrained('./model_save/300m', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    # model = model.to('cuda')

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")

    # # 3. cuda cache回调函数
    # my_trainer_callback = MyTrainerCallback()

    # # 4. 定义训练参数
    args = TrainingArguments(
        output_dir = config.output_dir,
        per_device_train_batch_size = config.per_device_train_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        num_train_epochs = config.num_train_epochs, 
        weight_decay = config.weight_decay,
        warmup_steps = config.warmup_steps,
        learning_rate = config.learning_rate,
        evaluation_strategy = config.evaluation_strategy,
        eval_steps = config.eval_steps,
        save_steps = config.save_steps,
        save_strategy = config.save_strategy,
        save_total_limit = config.save_total_limit,
        report_to = config.report_to,
        optim = config.optim,
        bf16 = config.bf16,
        fp16 = config.fp16,
        logging_steps = config.logging_steps,
        log_level = config.log_level,
        logging_first_step = config.logging_first_step,
        # group_by_length= config.group_by_length,
        # deepspeed='./ds_config_one_gpu.json',
    )

    trainer = PreTrainTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # callbacks=[my_trainer_callback],
    )

    # 5. 开始训练
    # `resume_from_checkpoint=True`参数可以从上次保存的检查点继续训练

    trainer.train(
        resume_from_checkpoint=config.resume_from_checkpoint,
    )

    # 6. 保存模型
    trainer.save_model(config.output_dir)

    if not eval_dataset:
        #  计算困惑度Perplexity 
        eval_results = trainer.evaluate()
        print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    # # 7. 最后保存训练的loss日志和模型
    loss_log = pd.DataFrame(trainer.state.log_history)

    if not os.path.exists(config.logs_dir):
        os.mkdir(config.logs_dir)
    loss_log.to_csv(f"{config.logs_dir}/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
