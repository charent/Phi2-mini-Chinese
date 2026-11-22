"""命令行入口：解析参数并分发到 pre / sft / dpo 三个阶段。

三种等价用法（重构后推荐前两种）::

    python -m phi2_mini --train_type pre ...
    phi2-mini --train_type pre ...
    python train_main.py --train_type pre ...   # 旧入口，下个大版本移除
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, cast

from phi2_mini.config import TrainConfig
from phi2_mini.logging_utils import get_logger

logger = get_logger(__name__)

TRAIN_TYPES = ('pre', 'sft', 'dpo')


def build_parser() -> Any:
    """构造 HfArgumentParser。

    transformers 在函数内导入：延迟导入可以避免仅使用 ``paths`` / ``config`` 时也加载
    重型依赖。``TrainingArguments`` 不是 dataclass，HfArgumentParser 的类型注解覆盖不到，
    因此这里做一次显式 cast。
    """
    from transformers import HfArgumentParser, TrainingArguments

    return HfArgumentParser(dataclass_types=cast(Any, (TrainConfig, TrainingArguments)))


def parse_arguments(argv: Sequence[str] | None = None) -> tuple[TrainConfig, Any]:
    """解析命令行，返回 ``(TrainConfig, TrainingArguments)``。"""
    parser = build_parser()
    config, train_args = parser.parse_args_into_dataclasses(args=list(argv) if argv is not None else None)
    return cast(TrainConfig, config), train_args


def dispatch(config: TrainConfig, train_args: Any) -> None:
    """按 ``train_type`` 分发到对应训练流程。"""
    if config.train_type not in TRAIN_TYPES:
        raise ValueError(f'`train_type` 必须是 {TRAIN_TYPES} 之一，当前为 {config.train_type!r}')

    match config.train_type:
        case 'pre':
            from phi2_mini.train.pretrain import pre_train

            pre_train(config, train_args)
        case 'sft':
            from phi2_mini.train.sft import sft_train

            sft_train(config, train_args)
        case 'dpo':
            from phi2_mini.train.dpo import dpo_train

            dpo_train(config, train_args)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行主函数；返回进程退出码。"""
    config, train_args = parse_arguments(argv)
    logger.info('自定义参数：%s', config.__dict__)
    dispatch(config, train_args)
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
