
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import os 

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, TrainingArguments, PhiForCausalLM
import pandas as pd
import time

from trl import DPOTrainer

from config import CustomArugments
from utils.utils import MyTrainerCallback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def split_prompt_and_responses(samples: dict[str, str], bos_token: str, eos_token: str) -> Dict[str, str]:
    
    prompts, chosens, rejects = [], [], []
    batch_size = len(samples['prompt'])

    # DPO 不用手动添加 EOS BOS，dpo trainer会自动加上，
    for i in range(batch_size):
        prompts.append(f"{samples['prompt'][i]}")
        chosens.append(f"{samples['chosen'][i]}")
        rejects.append(f"{samples['rejected'][i]}")

    return {
        'prompt': prompts,
        'chosen': chosens,
        'rejected':rejects,
    }


def get_dataset(file: str, map_fun_args: dict) -> Dataset:

    dataset = load_dataset(path='json', data_files=file, split='train', cache_dir='.cache')

    # 不添加 bos eos 就不map了
    # maped_dataset = dataset.map(split_prompt_and_responses, batched=True, fn_kwargs=map_fun_args, num_proc=4)

    return dataset

def dpo_train(cust_args: CustomArugments, train_args: TrainingArguments) -> None:

    # 0. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cust_args.pretrain_model_dir)
    print(f"vicab size: {len(tokenizer)}")

    # 1. 加载数据集
    map_fun_args = {
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
    }

    train_dataset = get_dataset(cust_args.train_files, map_fun_args)
    print(train_dataset)


    # 2. 加载模型
    # `model`和`model_ref`开始时是同一个模型，只训练`model`的参数，`model_ref`参数保存不变

    model = PhiForCausalLM.from_pretrained(cust_args.pretrain_model_dir)
    model_ref = PhiForCausalLM.from_pretrained(cust_args.pretrain_model_dir)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")


    # # 3. 定义训练中的回调函数
    # 清空cuda缓存，dpo要加载两个模型，显存占用较大，这能有效缓解低显存机器显存缓慢增长的问题
    empty_cuda_cahce = MyTrainerCallback()

    trainer = DPOTrainer(
        model,
        model_ref,
        args=train_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[empty_cuda_cahce],
        max_length=cust_args.max_seq_len,
        max_prompt_length=cust_args.max_seq_len
    )

    # 4. 训练
    trainer.train(
        resume_from_checkpoint=train_args.resume_from_checkpoint,
    )

    # 5. 保存模型
    trainer.save_model(train_args.output_dir)

    # 6. 最后保存训练的loss日志
    loss_log = pd.DataFrame(trainer.state.log_history)

    if not os.path.exists(cust_args.logs_dir):
        os.mkdir(cust_args.logs_dir)
    loss_log.to_csv(f"{cust_args.logs_dir}/dpo_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

