import os, platform, time, sys
from typing import Optional

import torch

from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling,Trainer, TrainingArguments, PhiForCausalLM, PhiConfig
from trl import DataCollatorForCompletionOnlyLM
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


instruction_template = "##提问:"
response_template = "##回答:"

def token_to_id(samples: dict[str, list], tokenizer: PreTrainedTokenizerFast, map_dtype: np.dtype, max_len: int, template_ids: tuple[list[int]]) -> dict:

    instruction_txt, response_txt = samples['instruction'], samples['output']
    instruction_ids = tokenizer(
        instruction_txt,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    response_ids = tokenizer(
        response_txt,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    

    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    ins_tmplate_ids, res_template_ids = template_ids

    input_ids = []

    # 在每个问答对的开始、结束添加bos、 eos，以及添加指令标记
    for inst, resp in zip(instruction_ids, response_ids):
        input_ids.append(
            [bos_id] + ins_tmplate_ids + inst + res_template_ids + resp + [eos_id]
        )

    # 转换为numpy  
    input_ids = [np.array(item, dtype=map_dtype) for item in input_ids]

    return {
            "input_ids": input_ids
        }

def get_maped_dataset(files: str|list[str], map_fun_args: dict) -> Dataset:
    '''
    获取数据集，请根据自己机器的性能更改 num_proc 的数量，以加快处理速度
    '''
    dataset = load_dataset(path='parquet', data_files=files, split='train', cache_dir='.cache')
    maped_dataset = dataset.map(token_to_id, batched=True, batch_size=2_0000, fn_kwargs=map_fun_args, remove_columns=dataset.column_names, num_proc=8)
    return maped_dataset


def sft_train(config: MyTrainArugment):

    # 0. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.train_from_model_dir)
    ins_tmplate_ids, res_template_ids = tokenizer.encode(instruction_template), tokenizer.encode(response_template)
    

    # 1. 加载数据集
    # token to id缓存到文件，使用的时候不用再次tokenize
    # 如果词表大小小于 65535 用uint16存储，节省磁盘空间，否则用uint32存储
    map_dtype = np.uint16 if len(tokenizer) < 65535 else np.uint32
    
    map_fun_args = {
        'tokenizer': tokenizer,
        'map_dtype': map_dtype,
        'max_len': config.max_len,
        'template_ids': (ins_tmplate_ids, res_template_ids),
    }

    train_dataset = get_maped_dataset(config.train_files, map_fun_args)
    if config.eval_file is not None:
        eval_dataset = get_maped_dataset(config.eval_file, map_fun_args)
    else:
        eval_dataset = None

    print(train_dataset, eval_dataset)


    # 2. 加载预训练模型
    model = PhiForCausalLM.from_pretrained(config.train_from_model_dir)

    # 如果配置了flash_attention_2，请手动设置set_default_dtype为float16
    #  Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes.
    if attn_implementation == 'flash_attention_2':
        torch.set_default_dtype(torch.bfloat16)

    # model = model.to_bettertransformer()

    # 另外一个使用flash_attention_2的方法
    # model = PhiForCausalLM.from_pretrained('./model_save/300m', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    # model = model.to('cuda')
    

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")

    
    # 3. 定义data_collator
    # `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型
    data_collator = DataCollatorForCompletionOnlyLM(instruction_template=ins_tmplate_ids, response_template=res_template_ids, tokenizer=tokenizer, mlm=False)

    # 4. cuda cache回调函数
    my_trainer_callback = MyTrainerCallback()

    # 5. 定义训练参数
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
        group_by_length = True,  # 按照长度排序，最长的最先训练，如果最长的都不会oom，后面也不会oom了
        # deepspeed='./ds_config_one_gpu.json',
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[my_trainer_callback],
    )

    # 6. 开始训练
    # `resume_from_checkpoint=True`参数可以从上次保存的检查点继续训练

    trainer.train(
        resume_from_checkpoint=config.resume_from_checkpoint,
    )

    # 保存模型
    trainer.save_model(config.output_dir)

    if not eval_dataset:
        #  计算困惑度Perplexity 
        eval_results = trainer.evaluate()
        print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    # # 7. 最后保存训练的loss日志和模型
    loss_log = pd.DataFrame(trainer.state.log_history)

    if not os.path.exists(config.logs_dir):
        os.mkdir(config.logs_dir)
    loss_log.to_csv(f"{config.logs_dir}/sft_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
