from os.path import dirname, abspath
from dataclasses import dataclass, field
# replace '\' on windows to '/'

PROJECT_ROOT: str = '/'.join(abspath(dirname(__file__)).split('\\')) if '\\' in abspath(dirname(__file__)) else abspath(dirname(__file__))


@dataclass
class CustomArugments:
    # 从['pre', 'sft', 'dpo']中选择一个，分别表示要进行预训练、sft微调、dpo优化
    train_type: str = field(default=None)

    pretrain_model_dir: str = field(default=None)

    # 训练数据集文件，可以有多个，文件只需要一个`txet`字段
    train_files: list[str] = field(default=None)

    # tokenizer保存的目录，一般和模型目录保持一致
    tokenizer_dir: str = field(default=None)

    # 评估数据集，可为空
    eval_file: str = field(default=None)

    # 训练的最长token_id长度（不是文本长度）
    max_seq_len: int = field(default=512)

    logs_dir: str = field(default='logs')