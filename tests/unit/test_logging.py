"""日志工具：同名 logger 只初始化一次，可选写文件。"""

from __future__ import annotations

import pytest

pytest.importorskip('colorlog')

from phi2_mini.logging_utils import default_log_file, get_logger


def test_get_logger_is_idempotent():
    logger = get_logger('phi2_mini_test_idempotent')
    handler_count = len(logger.handlers)

    again = get_logger('phi2_mini_test_idempotent')

    assert again is logger
    assert len(logger.handlers) == handler_count == 1


def test_get_logger_writes_file(tmp_path):
    log_file = tmp_path / 'nested' / 'run.log'
    logger = get_logger('phi2_mini_test_file', log_file=log_file)

    logger.info('hello log')
    for handler in logger.handlers:
        handler.flush()

    assert 'hello log' in log_file.read_text(encoding='utf-8')
    assert len(logger.handlers) == 2


def test_default_log_file_pattern():
    assert default_log_file('demo').name.startswith('demo-')
