from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_dirs() -> None:
    for directory in [
        DATA_DIR,
        OUTPUT_DIR,
        OUTPUT_DIR / "metrics",
        OUTPUT_DIR / "rankings",
        OUTPUT_DIR / "figures",
        OUTPUT_DIR / "representations",
        OUTPUT_DIR / "vocabularies",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
