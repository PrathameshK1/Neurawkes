from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def month_to_float(month_idx: int, within_month_rank: int, eps: float) -> float:
    # Ensures strictly increasing event times for multiple events in a month.
    return float(month_idx) + float(within_month_rank) * float(eps)


@dataclass(frozen=True)
class TimeRange:
    start: str  # YYYY-MM
    end: str    # YYYY-MM


