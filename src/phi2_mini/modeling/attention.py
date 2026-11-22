"""attention 后端选择。

重构前的写法是「Windows 上回落到 eager，其他平台保持 flash_attention_2」，
于是 Linux 机器没装 flash-attn 时仍然把 ``flash_attention_2`` 传给 PhiConfig，
模型初始化直接报错。这里改为「探测不到 flash-attn 一律回落 eager」。
"""

from __future__ import annotations

import importlib.util

FLASH_ATTENTION_2 = 'flash_attention_2'
EAGER = 'eager'


def flash_attn_installed() -> bool:
    """flash-attn 是否可导入（用 find_spec 避免真的加载 CUDA 模块）。"""
    return importlib.util.find_spec('flash_attn') is not None


def resolve_attn_implementation(flash_attn_available: bool | None = None) -> str:
    """返回可用的 ``attn_implementation``。

    Args:
        flash_attn_available: 显式指定是否可用（便于测试）；``None`` 时自动探测。
    """
    if flash_attn_available is None:
        flash_attn_available = flash_attn_installed()
    return FLASH_ATTENTION_2 if flash_attn_available else EAGER
