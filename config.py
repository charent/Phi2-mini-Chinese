from os.path import dirname, abspath
from dataclasses import dataclass, field
# replace '\' on windows to '/'

PROJECT_ROOT: str = '/'.join(abspath(dirname(__file__)).split('\\')) if '\\' in abspath(dirname(__file__)) else abspath(dirname(__file__))


@dataclass
class MyTrainArugment:
    # 从['pre', 'sft', 'dpo']中选择一个，分别表示要进行预训练、sft微调、dpo优化
    train_type: str

    # 训练数据集文件，可以有多个，文件只需要一个`txet`字段
    train_files: list[str] = field()

    # tokenizer保存的目录，一般和模型目录保持一致
    tokenizer_dir: str

    # 训练后的模型保存目录
    output_dir: str

    # sft、dpo阶段从哪个模型开始的模型目录，将会使用该目录内的tokenizer，设置的tokenizer_dir将会被忽略
    train_from_model_dir: str = None

    # 评估数据集，可为空
    eval_file: str = None

    # 练日志保存目录
    logs_dir: str = './logs'

    # 训练的最长token_id长度（不是文本长度）
    max_len: int = 512

    # 是否从最近的断点开始训练
    resume_from_checkpoint: bool = False
    
    bf16: bool = True
    fp16: bool = False
    eval_steps: int = 2000
    evaluation_strategy: str = 'steps'
    gradient_accumulation_steps: int = 32
    # group_by_length: bool = True
    learning_rate: float = 5e-4
    log_level: str = 'info'
    logging_first_step: bool = True
    logging_steps: int = 5
    num_train_epochs: int = 4
    optim: str = "adafactor"
    report_to: str = 'tensorboard'
    save_steps: int = 5000
    save_strategy: str = 'steps'
    save_total_limit: int = 10
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    per_device_train_batch_size: int = 4

    seed: int = 23333
