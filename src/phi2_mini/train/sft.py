"""SFT 指令微调流程。"""

from __future__ import annotations

from typing import Any, cast

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM

from phi2_mini.callbacks import DefaultTrainerCallback
from phi2_mini.config import TrainConfig
from phi2_mini.data.sft import INSTRUCTION_TEMPLATE, RESPONSE_TEMPLATE
from phi2_mini.data.sft import build_dataset as build_sft_dataset
from phi2_mini.logging_utils import get_logger
from phi2_mini.modeling.attention import resolve_attn_implementation
from phi2_mini.modeling.factory import load_phi_model
from phi2_mini.train.common import apply_default_dtype, log_model_size, report_perplexity, save_log_history

logger = get_logger(__name__)


def _build_completion_collator(
    tokenizer: PreTrainedTokenizerFast,
    instruction_template_ids: list[int],
    response_template_ids: list[int],
) -> DataCollatorForCompletionOnlyLM:
    """只在回答部分计算 loss。

    ``instruction_template`` 参数在部分 trl 版本中不存在，做一次兼容处理。
    """
    kwargs = {
        'response_template': response_template_ids,
        'tokenizer': tokenizer,
        'mlm': False,
    }
    try:
        return DataCollatorForCompletionOnlyLM(instruction_template=instruction_template_ids, **kwargs)
    except TypeError:  # pragma: no cover - 取决于 trl 版本
        return DataCollatorForCompletionOnlyLM(**kwargs)


def sft_train(config: TrainConfig, train_args: TrainingArguments) -> None:
    """在预训练产物上做指令微调。"""
    if not config.train_from_model_dir:
        raise ValueError('SFT 必须提供 --train_from_model_dir')
    if not config.train_files:
        raise ValueError('SFT 必须提供 --train_files')

    # 0. tokenizer 与模板 id
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.train_from_model_dir)
    instruction_template_ids = tokenizer.encode(INSTRUCTION_TEMPLATE)
    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE)

    # 1. 数据集
    train_dataset = build_sft_dataset(
        config.train_files,
        tokenizer,
        num_proc=config.num_proc,
        cache_dir=config.cache_dir,
    )
    eval_dataset = (
        build_sft_dataset(config.eval_file, tokenizer, num_proc=config.num_proc, cache_dir=config.cache_dir)
        if config.eval_file
        else None
    )
    logger.info('train=%s, eval=%s', train_dataset, eval_dataset)

    # 2. 模型
    attn_implementation = resolve_attn_implementation()
    logger.info('attention 实现：%s', attn_implementation)
    apply_default_dtype(attn_implementation)
    model = load_phi_model(config.train_from_model_dir, attn_implementation)
    log_model_size(model)

    # 3. Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=_build_completion_collator(tokenizer, instruction_template_ids, response_template_ids),
        train_dataset=cast('Any', train_dataset),
        eval_dataset=cast('Any', eval_dataset),
        callbacks=[DefaultTrainerCallback()],
    )

    # 4. 训练 / 保存
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.save_model(train_args.output_dir)

    # 5. 困惑度与 loss 日志
    report_perplexity(trainer, eval_dataset)
    save_log_history(trainer, config.logs_dir, 'sft_train')
