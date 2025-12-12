from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from nhp_torch.config.io import Config
from nhp_torch.data.events import EventDataset
from nhp_torch.data.load_csv import load_monthly_csv
from nhp_torch.data.splits import make_splits


@dataclass(frozen=True)
class BinnedCounts:
    periods: pd.PeriodIndex
    counts: np.ndarray  # (T, K) integer counts per month per type


def build_binned_counts(cfg: Config, ds: EventDataset) -> BinnedCounts:
    """Bin event stream into month buckets, per type count."""
    tbl = load_monthly_csv(
        cfg.paths["raw_csv"],
        date_column=cfg.data["date_column"],
        missing_value=float(cfg.data["missing_value"]),
    )
    periods = tbl.periods
    T = len(periods)
    K = len(ds.type_names)
    counts = np.zeros((T, K), dtype=np.int32)
    for mi, k in zip(ds.month_index, ds.types):
        if 0 <= mi < T:
            counts[mi, int(k)] += 1
    return BinnedCounts(periods=periods, counts=counts)


def split_indices(cfg: Config, b: BinnedCounts) -> dict[str, np.ndarray]:
    splits = make_splits(**cfg.splits)
    out: dict[str, np.ndarray] = {}
    for name, (a, z) in {"train": splits.train, "val": splits.val, "test": splits.test}.items():
        out[name] = np.where((b.periods >= a) & (b.periods <= z))[0]
    return out


class PoissonIID:
    """Homogeneous per-type Poisson baseline: λ_k constant over time."""

    def __init__(self, lam: np.ndarray) -> None:
        self.lam = lam  # (K,)

    @staticmethod
    def fit(counts: np.ndarray) -> "PoissonIID":
        lam = counts.mean(axis=0).astype(np.float64) + 1e-8
        return PoissonIID(lam=lam)

    def nll(self, counts: np.ndarray) -> float:
        # NLL for independent Poisson per bin (up to constant log y! term).
        y = counts.astype(np.float64)
        lam = self.lam[None, :]
        return float((lam - y * np.log(lam)).sum())


class HawkesAR1Poisson(torch.nn.Module):
    """Discrete-time 'Hawkes-like' baseline: λ_t = softplus(μ + A y_{t-1})."""

    def __init__(self, K: int) -> None:
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(K))
        self.A = torch.nn.Parameter(torch.zeros(K, K))
        self.softplus = torch.nn.Softplus()

    def forward(self, y_prev: torch.Tensor) -> torch.Tensor:
        # y_prev: (T, K) -> λ: (T, K)
        lin = self.mu[None, :] + y_prev @ self.A.T
        return self.softplus(lin) + 1e-8


@dataclass(frozen=True)
class HawkesAR1Result:
    model_state: dict[str, Any]
    best_val_nll: float


def fit_hawkes_ar1_poisson(
    *,
    counts: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    epochs: int = 1000,
    lr: float = 5e-2,
    weight_decay: float = 1e-3,
    patience: int = 100,
    seed: int = 7,
) -> HawkesAR1Result:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(counts, dtype=torch.float32, device=device)
    K = y.shape[1]
    model = HawkesAR1Poisson(K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # y_prev for t=0 is zero; model predicts λ_0 from mu only (fine).
    y_prev = torch.cat([torch.zeros(1, K, device=device), y[:-1]], dim=0)

    def nll_on(idxs: np.ndarray) -> torch.Tensor:
        lam = model(y_prev[idxs])
        yy = y[idxs]
        # Poisson NLL up to constant.
        return (lam - yy * torch.log(lam)).sum()

    best_val = float("inf")
    best_state: dict[str, Any] = {}
    bad = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = nll_on(idx_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val = float(nll_on(idx_val).item())
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    return HawkesAR1Result(model_state=best_state, best_val_nll=float(best_val))


def hawkes_ar1_nll_from_state(
    *,
    counts: np.ndarray,
    state: dict[str, Any],
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(counts, dtype=torch.float32, device=device)
    K = y.shape[1]
    model = HawkesAR1Poisson(K).to(device)
    model.load_state_dict({k: v.to(device) if hasattr(v, "to") else v for k, v in state.items()})
    model.eval()
    y_prev = torch.cat([torch.zeros(1, K, device=device), y[:-1]], dim=0)
    with torch.no_grad():
        lam = model(y_prev)
        nll = (lam - y * torch.log(lam)).sum()
    return float(nll.item())


