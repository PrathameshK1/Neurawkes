from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def events(self) -> dict[str, Any]:
        return self.raw["events"]

    @property
    def splits(self) -> dict[str, Any]:
        return self.raw["splits"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def train(self) -> dict[str, Any]:
        return self.raw["train"]

    @property
    def eval(self) -> dict[str, Any]:
        return self.raw["eval"]


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    return Config(raw=raw)


