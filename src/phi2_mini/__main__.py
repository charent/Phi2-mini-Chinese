"""``python -m phi2_mini`` 入口，等价于 ``phi2-mini`` 命令。"""

import sys

from phi2_mini.cli import main

if __name__ == '__main__':
    sys.exit(main())
