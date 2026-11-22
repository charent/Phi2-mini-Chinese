"""CLM 预训练流程。"""

from __future__ import annotations

from typing import Any, cast

from torch.utils.data import Sampler, SequentialSampler
from transformers import PhiForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments

from phi2_mini.config import TrainConfig
from phi2_mini.data.collator import EfficiencyMaskCollator
from phi2_mini.data.pretrain import build_dataset as build_pretrain_dataset
from phi2_mini.logging_utils import get_logger
from phi2_mini.modeling.attention import EAGER, FLASH_ATTENTION_2, resolve_attn_implementation
from phi2_mini.modeling.factory import build_phi_config
from phi2_mini.modeling.packing import patch_phi_attention_mask
from phi2_mini.train.common import apply_default_dtype, log_model_size, report_perplexity, save_log_history

logger = get_logger(__name__)


class PreTrainTrainer(Trainer):
    """预训练按原始顺序读数据（不做 shuffle）。"""

    def _get_train_sampler(self) -> Sampler | None:
        if self.train_dataset is None:  # pragma: no cover - Trainer 内部才会传 None
            return None
        return SequentialSampler(cast('Any', self.train_dataset))


def pre_train(config: TrainConfig, train_args: TrainingArguments) -> None:
    """从零初始化 Phi 模型并做因果语言模型预训练。"""
    if not config.tokenizer_dir:
        raise ValueError('预训练必须提供 --tokenizer_dir')
    if not config.train_files:
        raise ValueError('预训练必须提供 --train_files')

    # 0. tokenizer：len(tokenizer) 才包含 add_tokens 新增的 token
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)

    # 1. attention 后端与模型
    attn_implementation = resolve_attn_implementation()
    if attn_implementation == FLASH_ATTENTION_2:
        # doc packing 依赖 4D 块对角 mask，而 4.36.2 的 flash-attn 路径只接受 2D mask
        logger.warning('预训练使用 doc packing（4D attention mask），与 flash-attention-2 不兼容，改用 eager')
        attn_implementation = EAGER
    logger.info('attention 实现：%s', attn_implementation)
    patch_phi_attention_mask()
    apply_default_dtype(attn_implementation)

    phi_config, vocab_size = build_phi_config(tokenizer, attn_implementation)
    logger.info('tokenizer 词表大小：%d，取整后：%d', len(tokenizer), vocab_size)

    # 2. 数据集（token id 会缓存到磁盘，下次训练不必重新 tokenize）
    train_dataset = build_pretrain_dataset(
        config.train_files,
        tokenizer,
        config.max_seq_len,
        num_proc=config.num_proc,
        cache_dir=config.cache_dir,
    )
    eval_dataset = (
        build_pretrain_dataset(
            config.eval_file,
            tokenizer,
            config.max_seq_len,
            num_proc=config.num_proc,
            cache_dir=config.cache_dir,
        )
        if config.eval_file
        else None
    )
    logger.info('train=%s, eval=%s', train_dataset, eval_dataset)

    # 3. 模型
    model = PhiForCausalLM(phi_config)
    log_model_size(model)

    # 4. Trainer（collator 负责 doc 级 attention mask）
    trainer = PreTrainTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=EfficiencyMaskCollator(tokenizer),
        train_dataset=cast('Any', train_dataset),
        eval_dataset=cast('Any', eval_dataset),
    )

    # 5. 训练 / 保存
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.save_model(train_args.output_dir)

    # 6. 困惑度与 loss 日志
    report_perplexity(trainer, eval_dataset)
    save_log_history(trainer, config.logs_dir, 'pre_train')
