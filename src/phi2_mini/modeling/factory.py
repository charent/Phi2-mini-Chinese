"""Phi 模型配置构造与加载（预训练从零初始化，sft/dpo 从检查点加载）。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from transformers import PhiConfig, PhiForCausalLM, PreTrainedTokenizerFast

VOCAB_MULTIPLE = 64


def round_vocab_size(vocab_size: int, multiple: int = VOCAB_MULTIPLE) -> int:
    """把词表大小向上取整到 ``multiple`` 的整数倍，便于 CUDA 计算。"""
    if multiple <= 0:
        raise ValueError(f'multiple 必须为正整数，收到 {multiple}')
    if vocab_size % multiple == 0:
        return vocab_size
    return (vocab_size // multiple + 1) * multiple


def build_phi_config(
    tokenizer: PreTrainedTokenizerFast,
    attn_implementation: str,
    *,
    hidden_size: int = 1024,
    num_attention_heads: int = 16,
    num_hidden_layers: int = 24,
    intermediate_size: int = 4096,
    max_position_embeddings: int = 768,
    tie_word_embeddings: bool = False,
) -> tuple[PhiConfig, int]:
    """按 tokenizer 词表构造 ``PhiConfig``，返回 ``(config, 取整后的词表大小)``。

    词表大小注意：``len(tokenizer)`` 才包含 ``add_tokens`` 新增的 token，
    ``tokenizer.vocab_size`` 不包含。
    """
    from transformers import PhiConfig

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('tokenizer 必须设置 bos_token / eos_token')
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    vocab_size = round_vocab_size(len(tokenizer))
    config = PhiConfig(
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        max_position_embeddings=max_position_embeddings,
        intermediate_size=intermediate_size,
        attn_implementation=attn_implementation,
        tie_word_embeddings=tie_word_embeddings,
    )
    return config, vocab_size


def load_phi_model(
    model_dir: str | Path,
    attn_implementation: str | None = None,
) -> PhiForCausalLM:
    """从目录（或 HF 仓库 id）加载 ``PhiForCausalLM``。

    ``from_pretrained`` 在部分版本的类型注解里会返回 ``(model, kwargs)`` 元组，
    这里统一 cast 成模型本身。
    """
    from transformers import PhiForCausalLM

    if attn_implementation:
        model = PhiForCausalLM.from_pretrained(str(model_dir), attn_implementation=attn_implementation)
    else:
        model = PhiForCausalLM.from_pretrained(str(model_dir))
    return cast(PhiForCausalLM, model)
