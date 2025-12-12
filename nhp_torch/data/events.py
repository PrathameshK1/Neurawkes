from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nhp_torch.config.io import Config
from nhp_torch.data.load_csv import load_monthly_csv
from nhp_torch.data.splits import make_splits, slice_period
from nhp_torch.data.transforms import apply_zscore, compute_changes, fit_zstats
from nhp_torch.utils import month_to_float


@dataclass
class EventDataset:
    # Global timeline event stream (strictly increasing event times).
    times: list[float]
    types: list[int]   # event type index
    marks: list[float] # non-negative mark (e.g. abs(z))
    type_names: list[str]
    month_index: list[int]  # month integer index aligned to df rows (for debugging)
    split_mask: dict[str, list[bool]]  # keys: train/val/test per-event

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "times": self.times,
            "types": self.types,
            "marks": self.marks,
            "type_names": self.type_names,
            "month_index": self.month_index,
            "split_mask": self.split_mask,
        }

    def describe(self) -> str:
        n = len(self.times)
        counts = np.bincount(np.array(self.types, dtype=int), minlength=len(self.type_names))
        top = np.argsort(-counts)[:10]
        lines = [
            f"num_events={n}",
            f"num_types={len(self.type_names)}",
            "top_types=" + ", ".join(f"{self.type_names[i]}:{int(counts[i])}" for i in top if counts[i] > 0),
        ]
        for split in ("train", "val", "test"):
            m = np.array(self.split_mask[split], dtype=bool)
            lines.append(f"{split}_events={int(m.sum())}")
        return "\n".join(lines) + "\n"


def build_event_dataset(cfg: Config) -> EventDataset:
    tbl = load_monthly_csv(
        cfg.paths["raw_csv"],
        date_column=cfg.data["date_column"],
        missing_value=float(cfg.data["missing_value"]),
    )

    # Optional column filter: only keep columns matching any of the substrings (case-insensitive).
    col_filter: list[str] = list(cfg.data.get("column_filter", []))
    if col_filter:
        keep_cols = [c for c in tbl.df.columns if any(f.lower() in c.lower() for f in col_filter)]
        if not keep_cols:
            raise ValueError(f"column_filter matched no columns; check config. Filters: {col_filter}")
        tbl = type(tbl)(df=tbl.df[keep_cols], date_col=tbl.date_col)

    splits = make_splits(**cfg.splits)

    changes = compute_changes(tbl.df, mode=cfg.data["transform"])
    train_changes = slice_period(changes, *splits.train)

    stats = fit_zstats(train_changes)
    z = apply_zscore(changes, stats)

    tau = float(cfg.events["threshold_z"])
    eps = float(cfg.events["epsilon_time"])
    mark_mode = str(cfg.events["mark"])

    type_names: list[str] = []
    col_names = list(z.columns)
    for c in col_names:
        type_names.append(f"{c}__up")
        type_names.append(f"{c}__down")

    times: list[float] = []
    types: list[int] = []
    marks: list[float] = []
    month_index: list[int] = []

    # Map period -> sequential month index (0..T-1) for event timestamps.
    periods = list(tbl.periods)
    period_to_idx = {p: i for i, p in enumerate(periods)}

    # Iterate months; create deterministic within-month event ordering by column order.
    for p in periods:
        row = z.loc[p]
        idx_month = period_to_idx[p]
        within = 0
        for j, c in enumerate(col_names):
            zv = row[c]
            if pd.isna(zv):
                continue
            if zv >= tau:
                etype = 2 * j
                times.append(month_to_float(idx_month, within, eps))
                types.append(etype)
                marks.append(float(abs(zv) if mark_mode == "abs_z" else zv))
                month_index.append(idx_month)
                within += 1
            elif zv <= -tau:
                etype = 2 * j + 1
                times.append(month_to_float(idx_month, within, eps))
                types.append(etype)
                marks.append(float(abs(zv) if mark_mode == "abs_z" else -zv))
                month_index.append(idx_month)
                within += 1

    # Split masks (per-event) based on original period membership.
    train_start, train_end = splits.train
    val_start, val_end = splits.val
    test_start, test_end = splits.test

    def period_for_month_idx(mi: int) -> pd.Period:
        return periods[mi]

    split_mask = {"train": [], "val": [], "test": []}
    for mi in month_index:
        per = period_for_month_idx(mi)
        split_mask["train"].append(train_start <= per <= train_end)
        split_mask["val"].append(val_start <= per <= val_end)
        split_mask["test"].append(test_start <= per <= test_end)

    # Sanity: strictly increasing times.
    if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
        raise RuntimeError("Event times are not strictly increasing; check epsilon_time.")

    return EventDataset(
        times=times,
        types=types,
        marks=marks,
        type_names=type_names,
        month_index=month_index,
        split_mask=split_mask,
    )


