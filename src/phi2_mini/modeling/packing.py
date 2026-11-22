"""让 Phi 接受预构建的 4D（doc 级块对角）attention mask。

背景：预训练会把多个短 doc 拼成一条定长样本，需要「同一行内不同 doc 互不可见」的块对角
mask，这种结构只能用 4D mask 表达。但 transformers 4.36.2 的
``_prepare_4d_causal_attention_mask`` 只接受 2D mask（``AttentionMaskConverter.to_4d`` 里
直接 ``bsz, src_len = mask.size()``，传 4D 会抛 ``ValueError``）。

因此这里对 Phi 模块内的 mask 转换函数做一次「4D 直接透传」的包装：
- 4D：视为二进制可见性 mask（1 可见 / 0 屏蔽），转换成 attention 需要的**加性** mask；
- 2D / None：交给原始实现，行为不变。

幂等，且只影响 ``transformers.models.phi``，不影响其他模型。
"""

from __future__ import annotations

_PATCH_FLAG = '_phi2_mini_patched_4d_mask'


def _to_additive_mask(mask, dtype):
    """二进制 mask -> 加性 mask（可见位置 0，屏蔽位置 ``finfo.min``）。"""
    import torch

    return (1 - mask.to(dtype)) * torch.finfo(dtype).min


def patch_phi_attention_mask() -> bool:
    """给 Phi 打上 4D mask 透传补丁；返回本次是否真的执行了 patch。"""
    from transformers.models.phi import modeling_phi

    if getattr(modeling_phi, _PATCH_FLAG, False):
        return False

    original = modeling_phi._prepare_4d_causal_attention_mask

    def prepare_4d_causal_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        *args,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            return _to_additive_mask(attention_mask, inputs_embeds.dtype)
        return original(attention_mask, input_shape, inputs_embeds, past_key_values_length, *args, **kwargs)

    modeling_phi._prepare_4d_causal_attention_mask = prepare_4d_causal_attention_mask
    setattr(modeling_phi, _PATCH_FLAG, True)
    return True
