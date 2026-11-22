"""DPO 偏好优化流程。"""

from __future__ import annotations

from transformers import PreTrainedTokenizerFast, TrainingArguments
from trl import DPOTrainer

from phi2_mini.callbacks import EmptyCacheCallback
from phi2_mini.config import TrainConfig
from phi2_mini.data.dpo import load_dataset_file
from phi2_mini.logging_utils import get_logger
from phi2_mini.modeling.attention import resolve_attn_implementation
from phi2_mini.modeling.factory import load_phi_model
from phi2_mini.train.common import log_model_size, save_log_history

logger = get_logger(__name__)

DPO_BETA = 0.1


def dpo_train(config: TrainConfig, train_args: TrainingArguments) -> None:
    """用 prompt/chosen/rejected 数据对 SFT 模型做偏好优化。"""
    if not config.train_from_model_dir:
        raise ValueError('DPO 必须提供 --train_from_model_dir')
    if not config.train_files:
        raise ValueError('DPO 必须提供 --train_files')

    # 0. tokenizer（优先用独立 tokenizer 目录，缺省回落到模型目录）
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir or config.train_from_model_dir)
    logger.info('tokenizer 词表大小：%d', len(tokenizer))

    # 1. 数据集：DPO 由 trainer 自动补 bos/eos，不再手动拼接
    train_dataset = load_dataset_file(config.train_files, cache_dir=config.cache_dir)
    logger.info('train=%s', train_dataset)

    # 2. model 参与训练，model_ref 保持冻结
    attn_implementation = resolve_attn_implementation()
    logger.info('attention 实现：%s', attn_implementation)
    model = load_phi_model(config.train_from_model_dir, attn_implementation)
    model_ref = load_phi_model(config.train_from_model_dir, attn_implementation)
    log_model_size(model)

    # 3. Trainer：DPO 需要加载两份模型，显存压力大，定期清缓存
    trainer = DPOTrainer(
        model,
        model_ref,
        args=train_args,
        beta=DPO_BETA,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[EmptyCacheCallback()],
        max_length=config.max_seq_len,
        max_prompt_length=config.max_seq_len,
    )

    # 4. 训练 / 保存
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.save_model(train_args.output_dir)

    # 5. loss 日志
    save_log_history(trainer, config.logs_dir, 'dpo_train')
