from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from nhp_torch.config.io import Config
from nhp_torch.data.events import EventDataset
from nhp_torch.data.load_csv import load_monthly_csv
from nhp_torch.data.splits import make_splits
from nhp_torch.data.transforms import apply_zscore, compute_changes, fit_zstats


@dataclass(frozen=True)
class ShockDataset:
    X: np.ndarray  # (T, D) z-changes
    Y: np.ndarray  # (T, K) binary events per month per type
    periods: pd.PeriodIndex


def build_shock_dataset(cfg: Config, ds: EventDataset) -> ShockDataset:
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

    X = z.to_numpy(dtype=np.float32)
    T = X.shape[0]
    K = len(ds.type_names)
    Y = np.zeros((T, K), dtype=np.float32)
    for mi, k in zip(ds.month_index, ds.types):
        if 0 <= mi < T:
            Y[mi, int(k)] = 1.0
    return ShockDataset(X=X, Y=Y, periods=z.index)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
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


@dataclass(frozen=True)
class ClassifierResult:
    macro_auc: float | None


def fit_logistic_multilabel(
    *,
    cfg: Config,
    ds: EventDataset,
    hidden: int | None = None,
    epochs: int = 2000,
    lr: float = 2e-2,
    weight_decay: float = 1e-3,
    patience: int = 200,
) -> tuple[dict[str, torch.Tensor], ClassifierResult]:
    data = build_shock_dataset(cfg, ds)
    splits = make_splits(**cfg.splits)

    # Predict next-month events from current-month z (one-step).
    X = data.X[:-1]
    Y = data.Y[1:]
    periods = data.periods[1:]

    idx_train = np.where((periods >= splits.train[0]) & (periods <= splits.train[1]))[0]
    idx_val = np.where((periods >= splits.val[0]) & (periods <= splits.val[1]))[0]
    idx_test = np.where((periods >= splits.test[0]) & (periods <= splits.test[1]))[0]
    if idx_val.size == 0:
        # With short timelines and a one-step shift, it's possible to lose a full year.
        # Fall back to using train as a proxy for early stopping.
        idx_val = idx_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X, device=device)
    Yt = torch.tensor(Y, device=device)

    D = Xt.shape[1]
    K = Yt.shape[1]

    if hidden is None:
        model = torch.nn.Linear(D, K).to(device)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(D, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, K),
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = float("inf")
    best_state: dict[str, torch.Tensor] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xt[idx_train])
        loss = loss_fn(logits, Yt[idx_train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xt[idx_val]), Yt[idx_val]).item())
        if v < best:
            best = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    # Evaluate AUC macro on test.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xt[idx_test])).detach().cpu().numpy()
        labels = Yt[idx_test].detach().cpu().numpy()

    aucs = []
    for k in range(K):
        a = _auc(probs[:, k], labels[:, k])
        if a is not None:
            aucs.append(a)
    macro = float(np.mean(aucs)) if aucs else None
    return best_state, ClassifierResult(macro_auc=macro)


