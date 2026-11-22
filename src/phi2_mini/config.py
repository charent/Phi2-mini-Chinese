"""自定义训练参数。

与 ``transformers.TrainingArguments`` 一起由 ``HfArgumentParser`` 解析，因此字段
类型必须是 argparse 可识别的类型（``Optional[str]`` / ``Optional[List[str]]`` / 基本类型）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """本项目特有的训练参数（其余超参走 ``TrainingArguments``）。"""

    train_type: str | None = field(
        default=None,
        metadata={'help': "训练阶段，三选一：'pre'（预训练）/ 'sft'（指令微调）/ 'dpo'（偏好优化）"},
    )
    train_from_model_dir: str | None = field(
        default=None,
        metadata={'help': 'sft/dpo 的起始模型目录（预训练产物或上一阶段检查点）'},
    )
    train_files: list[str] | None = field(
        default=None,
        metadata={
            'help': '训练数据文件，可传多个；预训练/sft 为 parquet（需 text/instruction+output 列），dpo 为 json'
        },
    )
    tokenizer_dir: str | None = field(
        default=None,
        metadata={'help': 'tokenizer 目录，通常与模型目录保持一致'},
    )
    eval_file: str | None = field(
        default=None,
        metadata={'help': '评估数据文件，可为空'},
    )
    max_seq_len: int = field(
        default=512,
        metadata={'help': '单个样本的最大 token 数（不是文本字符数）'},
    )
    logs_dir: str = field(
        default='logs',
        metadata={'help': '训练 loss 日志（csv）输出目录'},
    )
    num_proc: int = field(
        default=8,
        metadata={'help': '数据集 map 的进程数，按机器性能调整；调试时建议设为 1'},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={'help': 'datasets 缓存目录，默认 <项目根>/.cache'},
    )


# 兼容重构前的类名（拼写有误），下一个版本移除
CustomArugments = TrainConfig
