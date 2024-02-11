
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import os 

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, TrainingArguments, PhiForCausalLM
import pandas as pd
import time

from trl import DPOTrainer

from config import MyTrainArugment
from utils.utils import MyTrainerCallback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def split_prompt_and_responses(samples: dict[str, str], bos_token: str, eos_token: str) -> Dict[str, str]:
    
    prompts, chosens, rejects = [], [], []
    batch_size = len(samples['prompt'])

    #添加bos eos
    for i in range(batch_size):
        prompts.append(f"{bos_token}{samples['prompt'][i]}{eos_token}")
        chosens.append(f"{bos_token}{samples['chosen'][i]}{eos_token}")
        rejects.append(f"{bos_token}{samples['rejected'][i]}{eos_token}")

    return {
        'prompt': prompts,
        'chosen': chosens,
        'rejected':rejects,
    }


def get_dataset(file: str, map_fun_args: dict) -> Dataset:

    dataset = load_dataset(path='json', data_files=file, split='train', cache_dir='.cache')
    maped_dataset = dataset.map(split_prompt_and_responses, batched=True, fn_kwargs=map_fun_args, num_proc=4)

    return maped_dataset

def dpo_train(config: MyTrainArugment) -> None:


    # 0. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.train_from_model_dir)
    print(f"vicab size: {len(tokenizer)}")

    # 1. 加载数据集
    map_fun_args = {
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
    }

    train_dataset = get_dataset(config.train_files, map_fun_args)
    print(train_dataset)


    # 2. 加载模型
    # `model`和`model_ref`开始时是同一个模型，只训练`model`的参数，`model_ref`参数保存不变

    model = PhiForCausalLM.from_pretrained(config.train_from_model_dir)
    model_ref = PhiForCausalLM.from_pretrained(config.train_from_model_dir)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")


    # # 3. 定义训练中的回调函数
    # 清空cuda缓存，dpo要加载两个模型，显存占用较大，这能有效缓解低显存机器显存缓慢增长的问题
    empty_cuda_cahce = MyTrainerCallback()


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
        remove_unused_columns=False,
        # group_by_length=True,
    )

    trainer = DPOTrainer(
        model,
        model_ref,
        args=args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[empty_cuda_cahce],
        max_length=config.max_len,
        max_prompt_length=config.max_len
    )


    # 4. 训练
    trainer.train(
        resume_from_checkpoint=config.resume_from_checkpoint,
    )

    # 5. 保存模型
    trainer.save_model(config.output_dir)

    # 6. 最后保存训练的loss日志
    loss_log = pd.DataFrame(trainer.state.log_history)

    if not os.path.exists(config.logs_dir):
        os.mkdir(config.logs_dir)
    loss_log.to_csv(f"{config.logs_dir}/dpo_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

