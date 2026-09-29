from __future__ import annotations

from pathlib import Path


def ensure_package_dirs():
    root = Path(__file__).resolve().parents[1]
    for d in [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "splits",
        root / "data" / "embeddings",
        root / "checkpoints" / "lesion",
        root / "checkpoints" / "patient",
        root / "checkpoints" / "temporal",
        root / "checkpoints" / "baselines",
        root / "reports" / "figures",
        root / "reports" / "tables",
        root / "reports" / "generated",
        root / "logs",
    ]:
        d.mkdir(parents=True, exist_ok=True)
