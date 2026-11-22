"""兼容旧入口：``python examples/web_demo.py``。

实现在 ``phi2_mini.inference.web_demo``；推荐直接运行
``python -m phi2_mini.inference.web_demo``。
"""

import sys
from pathlib import Path

try:
    from phi2_mini.inference.web_demo import main
except ModuleNotFoundError:  # pragma: no cover - 源码未安装时回落到 src 布局
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from phi2_mini.inference.web_demo import main

if __name__ == '__main__':
    main()
