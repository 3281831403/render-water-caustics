"""pytest 引导：确保仓库根目录在 sys.path 上，`import caustics` 可用。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
