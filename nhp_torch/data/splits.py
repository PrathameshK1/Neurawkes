from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplits:
    train: tuple[pd.Period, pd.Period]
    val: tuple[pd.Period, pd.Period]
    test: tuple[pd.Period, pd.Period]


def _to_period(ym: str) -> pd.Period:
    return pd.Period(ym, freq="M")


def make_splits(*, train_start: str, train_end: str, val_start: str, val_end: str, test_start: str, test_end: str) -> TimeSplits:
    return TimeSplits(
        train=(_to_period(train_start), _to_period(train_end)),
        val=(_to_period(val_start), _to_period(val_end)),
        test=(_to_period(test_start), _to_period(test_end)),
    )


def slice_period(df, start: pd.Period, end: pd.Period):
    return df.loc[(df.index >= start) & (df.index <= end)]


