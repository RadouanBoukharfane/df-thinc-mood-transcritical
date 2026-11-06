"""One-dimensional nitrogen-interface advection (Section 8.1, Table 3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nitrogen_interface_study import main  # noqa: E402

if __name__ == "__main__":
    main()
