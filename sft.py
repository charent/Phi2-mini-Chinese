# %%
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, PhiForCausalLM, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
import pandas as pd
import numpy as np
import time
import torch
from trl import DataCollatorForCompletionOnlyLM

# %% [markdown]
# # 1. 定义训练数据，tokenizer，预训练模型的路径及最大长度

# %%
sft_file = './data/sft_train_data.parquet'
tokenizer_dir = './model_save/tokenizer/'
sft_from_checkpoint_file = './model_save/pre/'
model_save_dir = './model_save/sft/'
max_seq_len = 320

# %% [markdown]
# # 2. 加载训练数据集

# %%
dataset = load_dataset(path='parquet', data_files=sft_file, split='train', cache_dir='.cache')

# %%
dataset

# %%
# samples = dataset[0:2]
# print(samples)

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
print(f"vicab size: {len(tokenizer)}")

# %% [markdown]
# ## 2.1 定义sft data_collator的指令字符
# 也可以手动将`instruction_template_ids`和`response_template_ids`添加到input_ids中的，因为如果是byte level tokenizer可能将`:`和后面的字符合并，导致找不到`instruction_template_ids`和`response_template_ids`。 
# 也可以像下文一样通过在`'#'`和`':'`前后手动加`'\n'`解决

# %%
instruction_template = "##提问:"
response_template = "##回答:"


# %%

map_dtype = np.uint16 if len(tokenizer) < 65535 else np.uint32

def batched_formatting_prompts_func(example: list[dict]) -> list[str]:
    batch_txt = []
    for i in range(len(example['instruction'])):
        text = f"{instruction_template}\n{example['instruction'][i]}\n{response_template}\n{example['output'][i]}[EOS]"
        batch_txt.append(text)

    # token to id 
    outputs = tokenizer(batch_txt, return_attention_mask=False)
    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

    return {
            "input_ids": input_ids
        }

# print(batched_formatting_prompts_func(samples))

# %%
dataset = dataset.map(batched_formatting_prompts_func, batched=True, remove_columns=dataset.column_names).shuffle(23333)

# %% [markdown]
# ## 2.2 定义data_collator

# %%
# mlm=False表示训练的是CLM模型
data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

# %% [markdown]
# # 4. 加载预训练模型

# %%

model = PhiForCausalLM.from_pretrained(sft_from_checkpoint_file)

model_size = sum(t.numel() for t in model.parameters())
print(f"Phi2 size: {model_size / 1000**2:.2f}M parameters")

# %% [markdown]
# ## 定义训练过程中的回调函数
# N次log之后情况cuda缓存，能有效缓解低显存机器显存缓慢增长的问题

# %%
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()
            
empty_cuda_cahce = EmptyCudaCacheCallback()

# %% 
my_datasets =  dataset.train_test_split(test_size=4096)

# %% [markdown]
# # 5. 定义训练参数

# %%
args = TrainingArguments(
    output_dir=model_save_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=5e-5,
    evaluation_strategy='steps',
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=3,
    report_to='tensorboard',
    optim="adafactor",
    bf16=True,
    logging_steps=10,
    log_level='info',
    logging_first_step=True,
    group_by_length=True,
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=my_datasets['train'],
    eval_dataset=my_datasets['test'],
    callbacks=[empty_cuda_cahce],
)


# %% [markdown]
# # 6. 开始训练

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
# # 7. 保存日志和模型

# %%
loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/sft_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(model_save_dir)

# %%



