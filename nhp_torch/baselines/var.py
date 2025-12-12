from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhp_torch.config.io import Config
from nhp_torch.data.load_csv import load_monthly_csv
from nhp_torch.data.splits import make_splits
from nhp_torch.data.transforms import apply_zscore, compute_changes, fit_zstats


@dataclass(frozen=True)
class Var1Model:
    A: np.ndarray  # (D, D)
    b: np.ndarray  # (D,)
    columns: list[str]

    def predict_next(self, x_t: np.ndarray) -> np.ndarray:
        return x_t @ self.A.T + self.b[None, :]


def fit_var1(cfg: Config) -> tuple[pd.PeriodIndex, pd.DataFrame, Var1Model]:
    """Fit VAR(1) on z-scored monthly changes (train only)."""
    tbl = load_monthly_csv(
        cfg.paths["raw_csv"],
        date_column=cfg.data["date_column"],
        missing_value=float(cfg.data["missing_value"]),
    )
    splits = make_splits(**cfg.splits)
    changes = compute_changes(tbl.df, mode=cfg.data["transform"])
    train_changes = changes.loc[(changes.index >= splits.train[0]) & (changes.index <= splits.train[1])]
    stats = fit_zstats(train_changes)
    z = apply_zscore(changes, stats)

    # Build X_t -> X_{t+1}, dropping rows with NaNs.
    X = z.to_numpy(dtype=np.float64)
    X_t = X[:-1]
    X_next = X[1:]
    mask = np.isfinite(X_t).all(axis=1) & np.isfinite(X_next).all(axis=1)
    X_t = X_t[mask]
    X_next = X_next[mask]

    # Ridge regression closed-form for stability.
    lam = 1e-2
    D = X_t.shape[1]
    XtX = X_t.T @ X_t + lam * np.eye(D)
    XtY = X_t.T @ X_next
    A = np.linalg.solve(XtX, XtY).T  # (D,D)
    b = X_next.mean(axis=0) - X_t.mean(axis=0) @ A.T
    return tbl.periods, z, Var1Model(A=A, b=b, columns=list(z.columns))


def eval_var1_nextmonth_auc(
    *,
    cfg: Config,
    ds: "EventDataset",
) -> float | None:
    """Macro AUC (mean over types) for next-month event prediction using VAR(1) on z-changes.

    Score for type (col_j__up) = zhat_j
    Score for type (col_j__down) = -zhat_j
    """
    from nhp_torch.baselines.shock_classifier import build_shock_dataset, _auc  # reuse AUC impl
    from nhp_torch.data.splits import make_splits

    periods, z, model = fit_var1(cfg)
    data = build_shock_dataset(cfg, ds)
    splits = make_splits(**cfg.splits)

    X = data.X[:-1].astype(np.float64)
    Y = data.Y[1:].astype(np.float64)
    per = data.periods[1:]
    idx_test = np.where((per >= splits.test[0]) & (per <= splits.test[1]))[0]
    if idx_test.size == 0:
        return None

    # Predict z_{t} for t in idx_test from X_{t-1}.
    zhat = model.predict_next(X)[idx_test]  # (Ntest, D)
    labels = Y[idx_test]  # (Ntest, K)
    K = labels.shape[1]

    # Build scores for each type.
    # Types are in ds.type_names: [col0__up, col0__down, col1__up, col1__down, ...]
    aucs = []
    for k in range(K):
        j = k // 2
        score = zhat[:, j] if (k % 2 == 0) else -zhat[:, j]
        a = _auc(score.astype(np.float64), labels[:, k].astype(np.int32))
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else None


