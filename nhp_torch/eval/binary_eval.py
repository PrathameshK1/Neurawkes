from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nhp_torch.data.events import EventDataset


@dataclass(frozen=True)
class BinaryEvalResult:
    horizon_months: int
    per_type_auc: dict[str, float | None]
    macro_auc_mean: float | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "horizon_months": self.horizon_months,
            "per_type_auc": self.per_type_auc,
            "macro_auc_mean": self.macro_auc_mean,
        }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    # Basic ROC AUC without sklearn, requires both classes present.
    labels = labels.astype(np.int32)
    if labels.min() == labels.max():
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = len(labels) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def month_event_matrix(ds: EventDataset) -> np.ndarray:
    T = max(ds.month_index) + 1 if ds.month_index else 0
    K = len(ds.type_names)
    y = np.zeros((T, K), dtype=np.int32)
    for mi, k in zip(ds.month_index, ds.types):
        y[mi, int(k)] = 1
    return y


def eval_horizon_auc(
    *,
    model,
    ds: EventDataset,
    split: str,
    horizon_months: int,
    grid_points_per_month: int = 4,
) -> BinaryEvalResult:
    """Evaluate AUC for predicting whether each type occurs within next H months.

    Score: p = 1 - exp(-∫_t^{t+H} λ_k(u) du) using a grid approximation.
    Labels: whether any event of type k occurs in (t, t+H].
    """
    device = next(model.parameters()).device
    y = month_event_matrix(ds)  # (T,K) 0/1 per month
    T, K = y.shape

    # Determine which months belong to split using per-event mask, then map to month range.
    # We approximate by using the min/max month indices of events in split.
    ev_mask = np.array(ds.split_mask[split], dtype=bool)
    months_in_split = np.array(ds.month_index, dtype=int)[ev_mask]
    if months_in_split.size == 0:
        return BinaryEvalResult(horizon_months=horizon_months, per_type_auc={n: None for n in ds.type_names}, macro_auc_mean=None)
    m_start = int(months_in_split.min())
    m_end = int(months_in_split.max())

    # Use full history events for intensity path queries.
    times_all = torch.tensor(ds.times, dtype=torch.float32, device=device)
    types_all = torch.tensor(ds.types, dtype=torch.long, device=device)
    marks_all = torch.tensor(ds.marks, dtype=torch.float32, device=device)

    # For each month boundary t=m, query intensities on grid in [m, m+H].
    scores = [[] for _ in range(K)]
    labels = [[] for _ in range(K)]
    with torch.no_grad():
        for m in range(m_start, m_end + 1):
            t0 = float(m)
            t1 = float(min(m + horizon_months, T))
            if t1 <= t0:
                continue
            grid = torch.linspace(
                t0,
                t1,
                steps=max(2, int((t1 - t0) * grid_points_per_month) + 1),
                device=device,
            )
            lam = model.forward_intensity_path(times_all, types_all, marks_all, query_times=grid)  # (G,K)
            integ = torch.trapezoid(lam, grid, dim=0)  # (K,)
            prob = 1.0 - torch.exp(-integ)
            prob_np = prob.detach().cpu().numpy()

            # Label: any event in months (m+1..m+H) inclusive.
            end_m = min(m + horizon_months, T - 1)
            future = y[m + 1 : end_m + 1]
            lab = future.max(axis=0) if future.size else np.zeros((K,), dtype=np.int32)

            for k in range(K):
                scores[k].append(float(prob_np[k]))
                labels[k].append(int(lab[k]))

    per_type_auc: dict[str, float | None] = {}
    aucs: list[float] = []
    for k, name in enumerate(ds.type_names):
        s = np.array(scores[k], dtype=np.float64)
        l = np.array(labels[k], dtype=np.int32)
        a = _auc(s, l)
        per_type_auc[name] = a
        if a is not None:
            aucs.append(a)

    macro = float(np.mean(aucs)) if aucs else None
    return BinaryEvalResult(horizon_months=horizon_months, per_type_auc=per_type_auc, macro_auc_mean=macro)


