"""项目内路径的统一入口。

所有路径都基于「仓库根目录」解析，而不是当前工作目录，因此训练脚本可以在任意
cwd 下运行（重构前依赖 ``sys.path.extend(['.', '..'])`` 与脚本里 ``cd ..``）。

本项目通过 ``uv sync`` 以 editable 方式安装，所以 ``PROJECT_ROOT`` 指向源码仓库根目录。
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'model_save'
LOG_DIR = PROJECT_ROOT / 'logs'
CACHE_DIR = PROJECT_ROOT / '.cache'


def resolve_path(value: str | Path | None, default: Path) -> str:
    """用户显式传入的路径优先，否则回落到默认路径。

    返回 ``str``，便于直接传给 transformers / datasets 等第三方 API。
    """
    return str(value) if value else str(default)
