"""Small utilities that keep scientific artifacts independent of wall time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_scientific_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write deterministic scientific content and reject runtime metadata."""

    forbidden = {key for key in summary if "runtime" in key.lower() or "wall" in key.lower()}
    if forbidden:
        raise ValueError(f"timing fields are forbidden in scientific summary: {sorted(forbidden)}")
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_execution_metadata(path: Path, *, runtime_seconds: float, **metadata: Any) -> None:
    """Write explicitly non-scientific execution metadata."""

    if runtime_seconds < 0.0:
        raise ValueError("runtime must be nonnegative")
    payload = {"runtime_seconds": float(runtime_seconds), **metadata}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
