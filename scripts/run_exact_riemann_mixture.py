"""Print the exact Riemann solutions of the nitrogen/n-dodecane shock tubes (Section 8.3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import runpy  # noqa: E402

if __name__ == "__main__":
    runpy.run_module("exact_riemann_mixture", run_name="__main__")
