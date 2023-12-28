# %%
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, PhiConfig, PhiForCausalLM, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import pandas as pd
import time
import torch

# %% 
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# %% [markdown]
# # 1. 数据来源，保存路径，最大长度定义

# %%
tokenizer_dir = './model_save/tokenizer/'
model_save_dir = './model_save/pre/'
logs_dir = './logs/'
train_files = ['./data/wiki_chunk_320_2.2M.parquet', './data/bell_pretrain_3M.parquet']
max_seq_len = 512

# %% [markdown]
# # 2. 加载训练好的tokenizer
# 如果你使用的`add_tokens`方法添加了自己的token，必须要用`len(tokenizer)`获取长度，`tokenizer.vocab_size`统计不包含你添加的字符。

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
print(f"vicab size: {len(tokenizer)}")

# %% [markdown]
# # 3. 加载数据集

# %%
dataset = load_dataset(path='parquet', data_files=train_files, split='train', cache_dir='.cache')

# %%
dataset

# %%
# samples = dataset['text'][0:5]
# print(samples)

# %% [markdown]
# ## token to id缓存到文件，使用的时候不用再次tokenize

# %%
def token_to_id(samples: dict[str, list]) -> dict:
    # batch_txt = []
    # for txt in samples['text']:
    #     batch_txt.append(
    #         f"[BOS]{txt}[EOS]"
    #     )

    batch_txt = samples['text']
    outputs = tokenizer(
        batch_txt,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    return {
        "input_ids": outputs["input_ids"], 
        }

# print(token_to_id({'text':['判断给定的文章是否符合语法规则。如果不符合，请提供修改建议。\n','下面是一篇文章的开头: "为了探讨这个主题，本文将提供一系列数据和实例，以证明这一观点。']}))


# %%

tokenized_datasets = dataset.map(
    token_to_id, batched=True, batch_size=1_0000, remove_columns=dataset.column_names
)
tokenized_datasets

# %% [markdown]
# # 4. 定义data_collator
# `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型

# %%
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# %%
# few_data = [tokenized_datasets[i] for i in range(5)]
# print(few_data)

# %% [markdown]
# ##  验证一下数据看padding、输入输出是否符合要求

# %%
# out = data_collator(few_data)
# print(out.keys())
# for key in out:
#     # print(out[key])
#     print(f"{key} shape: {out[key].shape}")

# # input_ids 和 labels 相同
# sum(out['input_ids'][0] == out['labels'][0]) == sum(out['attention_mask'][0])

# %% [markdown]
# # 5. 定义模型
# 从`config`定义，不是`from_pretrained`。 
# 为了方便cuda计算，词表的大小注意一下，如果不是64的整数倍，可以手动向上取整为64的整数倍，也可以是其他 $2^x$ 数值的整数倍，如32、128、256都行。

# %%
vocab_size = len(tokenizer)
if vocab_size % 64 != 0:
    vocab_size = (vocab_size // 64 + 1) * 64
print(f"final vocab sieze: {vocab_size}")

# %%
phi_config = PhiConfig(
    vocab_size=vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=24,
    max_position_embeddings=512,
    intermediate_size=4096,
)

model = PhiForCausalLM(phi_config)
# model = model.to_bettertransformer()

model_size = sum(t.numel() for t in model.parameters())
print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")

# %% [markdown]
# # 6. cuda cache回调函数

# %%
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()
            
empty_cuda_cahce = EmptyCudaCacheCallback()

# %% [markdown]
# # 6. 定义训练参数

# %%
args = TrainingArguments(
    output_dir=model_save_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=4,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=5e-4,
    save_steps=2000,
    save_total_limit=3,
    report_to='tensorboard',
    optim="adafactor",
    bf16=True,
    logging_steps=10,
    log_level='info',
    logging_first_step=True,
    # deepspeed='./ds_config_one_gpu.json',
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
    callbacks=[empty_cuda_cahce],
)

# %% [markdown]
# # 7. 开始训练
# `resume_from_checkpoint=True`参数可以从上次保存的检查点继续训练

# %%
trainer.train(
    # resume_from_checkpoint=True
)

# %% [markdown]
# 

# %% [markdown]
# # 8. 最后保存训练的loss日志和模型

# %%

loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(model_save_dir)


