"""三个训练阶段共用的收尾逻辑：dtype、日志、困惑度。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import Trainer

from phi2_mini.logging_utils import get_logger
from phi2_mini.modeling.attention import FLASH_ATTENTION_2
from phi2_mini.paths import LOG_DIR

logger = get_logger(__name__)


def apply_default_dtype(attn_implementation: str) -> None:
    """Flash Attention 2 只支持 float16 / bfloat16。"""
    if attn_implementation == FLASH_ATTENTION_2:
        torch.set_default_dtype(torch.bfloat16)


def ensure_dir(path: str | Path) -> Path:
    """``mkdir -p``，返回 Path 便于链式使用。"""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def log_model_size(model: Any) -> int:
    """打印参数量并返回。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info('Phi-2 参数量：%.1fM', total / 1000**2)
    return total


def save_log_history(trainer: Trainer, logs_dir: str | Path, prefix: str) -> Path:
    """把 ``trainer.state.log_history`` 保存为 csv；返回文件路径。"""
    target_dir = ensure_dir(logs_dir or LOG_DIR)
    history = pd.DataFrame(trainer.state.log_history)
    output = target_dir / f'{prefix}_{time.strftime("%Y%m%d-%H%M")}.csv'
    history.to_csv(output, index=False)
    logger.info('训练日志已保存：%s', output)
    return output


def report_perplexity(trainer: Trainer, eval_dataset: Any) -> float | None:
    """有 eval 集时计算困惑度。

    重构前是 ``if not eval_dataset: trainer.evaluate()``：条件写反，且
    ``eval_dataset=None`` 时 ``evaluate()`` 会直接抛错。改为「存在 eval 集才算」。
    """
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.info('未提供 eval 数据集，跳过 perplexity 计算')
        return None

    results = trainer.evaluate()
    loss = results.get('eval_loss')
    if loss is None:
        logger.warning('evaluate() 未返回 eval_loss，跳过 perplexity 计算')
        return None

    perplexity = float(np.exp(loss))
    logger.info('Perplexity: %.2f', perplexity)
    return perplexity
