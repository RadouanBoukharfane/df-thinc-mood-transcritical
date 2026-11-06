"""Two-dimensional n-dodecane drop, split solver (Section 8.5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drop_euler_2d import main  # noqa: E402

if __name__ == "__main__":
    main()
