from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZStats:
    mean: pd.Series
    std: pd.Series


def compute_changes(df: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    """Compute per-column monthly changes.

    - mode="logdiff": log(x_t) - log(x_{t-1}) for x>0 else NaN
    - mode="pct": (x_t / x_{t-1}) - 1 for x_{t-1}!=0 else NaN
    """
    if mode not in {"logdiff", "pct"}:
        raise ValueError(f"Unsupported transform: {mode!r}")

    if mode == "logdiff":
        safe = df.where(df > 0.0)
        return np.log(safe).diff()

    # pct
    denom = df.shift(1)
    safe = denom.where(denom != 0.0)
    return (df / safe) - 1.0


def fit_zstats(changes: pd.DataFrame) -> ZStats:
    mean = changes.mean(axis=0, skipna=True)
    std = changes.std(axis=0, ddof=0, skipna=True).replace(0.0, np.nan)
    return ZStats(mean=mean, std=std)


def apply_zscore(changes: pd.DataFrame, stats: ZStats) -> pd.DataFrame:
    return (changes - stats.mean) / stats.std


