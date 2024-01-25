# %%
import os, platform, time
from typing import Optional

from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, PhiConfig, PhiForCausalLM, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, Dataset
import pandas as pd
from transformers.trainer_callback import TrainerControl, TrainerState
import numpy as np
from dataclasses import dataclass,field
import torch

# %%
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


attn_implementation = 'flash_attention_2'
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_implementation = 'eager'

# %% [markdown]
# # 1. 训练数据来源

TRAIN_FILES = [
    './data/wiki_chunk_320_2.2M.parquet', 
    './data/bell_pretrain_400_3M.parquet',
]

EVAL_FILE = './data/pretrain_eval_400_1w.parquet'

# %%

@dataclass
class PretrainArguments:
    tokenizer_dir: str = './model_save/tokenizer/'
    model_save_dir: str = './model_save/pre/'
    logs_dir: str = './logs/'
    train_files: list[str] = field(default_factory=lambda: TRAIN_FILES)
    eval_file: str = EVAL_FILE
    max_seq_len: int = 512

    # Windows 使用默认的attention实现，
    attn_implementation: str = 'eager' if platform.system() == 'Windows' else attn_implementation


pretrain_args = PretrainArguments()

# %% [markdown]
# # 2. 加载训练好的tokenizer
# 如果你使用的`add_tokens`方法添加了自己的token，必须要用`len(tokenizer)`获取长度，`tokenizer.vocab_size`统计不包含你添加的字符。

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrain_args.tokenizer_dir)

# %% [markdown]
# # 5. 定义模型
# 从`config`定义，不是`from_pretrained`。 
# 为了方便cuda计算，词表的大小注意一下，如果不是64的整数倍，可以手动向上取整为64的整数倍，也可以是其他 $2^x$ 数值的整数倍，如32、128、256都行。

# %%
vocab_size = len(tokenizer)
if vocab_size % 64 != 0:
    vocab_size = (vocab_size // 64 + 1) * 64
print(f"source vocab size: {len(tokenizer)}, final vocab sieze: {vocab_size}")
 
# %% [markdown]
# ## token to id缓存到文件，使用的时候不用再次tokenize
# 如果词表大小小于 65535 用uint16存储，节省磁盘空间，否则用uint32存储
# %%
map_dtype = np.uint16 if vocab_size < 65535 else np.uint32

def token_to_id(samples: dict[str, list]) -> dict:

    batch_txt = samples['text']
    outputs = tokenizer(
        batch_txt,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

    return {
            "input_ids": input_ids
        }


# step 3 加载数据集

# %%
def get_maped_dataset(files: str|list[str]) -> Dataset:
    dataset = load_dataset(path='parquet', data_files=files, split='train', cache_dir='.cache')
    maped_dataset = dataset.map(token_to_id, batched=True, batch_size=1_0000, remove_columns=dataset.column_names)
    return maped_dataset

train_dataset = get_maped_dataset(pretrain_args.train_files)
eval_dataset = get_maped_dataset(pretrain_args.eval_file)

print(train_dataset, eval_dataset)
# %% [markdown]
# # 4. 定义data_collator
# `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型

# %%
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# %%
# 如果配置了flash_attention_2，请手动设置set_default_dtype为float16
#  Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes.
if pretrain_args.attn_implementation == 'flash_attention_2':
    torch.set_default_dtype(torch.bfloat16)


# %%
phi_config = PhiConfig(
    vocab_size=vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    hidden_size=960,
    num_attention_heads=16,
    num_hidden_layers=24,
    max_position_embeddings=512,
    intermediate_size=4096,
    attn_implementation=pretrain_args.attn_implementation,
)

model = PhiForCausalLM(phi_config)
# model = model.to_bettertransformer()

# 另外一个使用flash_attention_2的方法
# model = PhiForCausalLM.from_pretrained('./model_save/300m', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
# model = model.to('cuda')

model_size = sum(t.numel() for t in model.parameters())
print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")
# %% [markdown]
# # 6. cuda cache回调函数

# %%
class MyTrainerCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM
        '''
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        在on_epoch_end时保存一次模型。
        TrainingArguments的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔 save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近3个检查点。
        '''
        # 设置should_save=True并返回即可
        control.should_save = True
        return control
    
my_trainer_callback = MyTrainerCallback()

# %% [markdown]
# # 6. 定义训练参数

# %%
args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    num_train_epochs=4,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=5e-4,
    evaluation_strategy='steps',
    eval_steps=2000,
    save_steps=2000,
    save_strategy='steps',
    save_total_limit=3,
    report_to='tensorboard',
    optim="adafactor",
    bf16=True,
    logging_steps=5,
    log_level='info',
    logging_first_step=True,
    # group_by_length=True,
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

# %% [markdown]
# # 7. 开始训练
# `resume_from_checkpoint=True`参数可以从上次保存的检查点继续训练

# %%
trainer.train(
    # resume_from_checkpoint=True
)

# %% [markdown]
#  计算困惑度Perplexity 

# %%
eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

# %% [markdown]
# # 8. 最后保存训练的loss日志和模型

# %%

loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(pretrain_args.model_save_dir)


