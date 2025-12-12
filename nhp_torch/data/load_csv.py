from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoadedTable:
    df: pd.DataFrame  # index: Period[M], columns: features (float)
    date_col: str

    @property
    def periods(self) -> pd.PeriodIndex:
        return self.df.index  # type: ignore[return-value]


def load_monthly_csv(
    path: str | Path,
    *,
    date_column: str,
    missing_value: float = -1.0,
) -> LoadedTable:
    p = Path(path)
    df = pd.read_csv(p)
    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column!r}")

    # Parse YYYY-MM into monthly period index.
    periods = pd.PeriodIndex(pd.to_datetime(df[date_column], format="%Y-%m"), freq="M")
    df = df.drop(columns=[date_column])

    # Numeric conversion; mark missing_value as NaN.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(missing_value, np.nan)
    df.index = periods
    df = df.sort_index()

    return LoadedTable(df=df, date_col=date_column)


