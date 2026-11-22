"""兼容旧入口：``python train_main.py ...`` 等价于 ``python -m phi2_mini ...``。

新代码请直接使用 ``python -m phi2_mini``（或安装后的 ``phi2-mini`` 命令），
该文件将在下个大版本移除。
"""

import sys
from pathlib import Path

try:
    from phi2_mini.cli import main
except ModuleNotFoundError:  # pragma: no cover - 源码未安装时回落到 src 布局
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    from phi2_mini.cli import main

if __name__ == '__main__':
    sys.exit(main())
