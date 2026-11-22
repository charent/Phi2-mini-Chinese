"""命令行集成测试：以子进程方式运行，验证「与 cwd 无关」且参数校验可用。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('transformers')

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'


def run_python(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    env['PYTHONIOENCODING'] = 'utf-8'
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        # Windows 默认按 GBK 解码子进程输出，含中文的 help 文本会直接崩掉
        encoding='utf-8',
        errors='replace',
        timeout=180,
        check=False,
    )


def test_help_lists_custom_arguments(tmp_path):
    result = run_python(['-m', 'phi2_mini', '--help'], tmp_path)

    assert result.returncode == 0, result.stderr
    assert '--train_type' in result.stdout
    assert '--train_files' in result.stdout


def test_invalid_train_type_exits_nonzero(tmp_path):
    result = run_python(
        ['-m', 'phi2_mini', '--train_type', 'xxx', '--output_dir', str(tmp_path / 'out')],
        tmp_path,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert 'pre' in output and 'sft' in output and 'dpo' in output


def test_missing_train_type_exits_nonzero(tmp_path):
    result = run_python(['-m', 'phi2_mini', '--output_dir', str(tmp_path / 'out')], tmp_path)

    assert result.returncode != 0


def test_legacy_entry_script_still_works(tmp_path):
    result = run_python([str(ROOT / 'train_main.py'), '--help'], tmp_path)

    assert result.returncode == 0, result.stderr
    assert '--train_type' in result.stdout
