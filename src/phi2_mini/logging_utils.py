"""基于 colorlog 的轻量日志封装。

重构前每个模块各自 ``print``、``utils/logger.py`` 又从未被引用；这里统一为
``get_logger()`` 工厂，并修复原实现「每次实例化都向同名 logger 追加 handler、
导致同一条日志重复输出」的问题。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import colorlog

from phi2_mini.paths import LOG_DIR

LOG_FORMAT = '[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


def get_logger(
    name: str = 'phi2_mini',
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """获取（并幂等初始化）一个带彩色控制台输出的 logger。

    同名 logger 只会初始化一次，重复调用返回同一实例；需要写文件时传入 ``log_file``。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console = colorlog.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(colorlog.ColoredFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT, log_colors=LOG_COLORS))
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode='a', encoding='utf-8')
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    return logger


def default_log_file(name: str) -> Path:
    """按 ``<项目根>/logs/<name>-YYYYMMDD.log`` 规则生成日志文件路径。"""
    import time

    return LOG_DIR / f'{name}-{time.strftime("%Y%m%d")}.log'
