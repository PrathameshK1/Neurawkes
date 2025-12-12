from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nhp_torch.data.events import EventDataset


def month_event_matrix(ds: EventDataset) -> np.ndarray:
    """Binary matrix Y[month, type] = 1 if an event of that type occurs in that month."""
    T = max(ds.month_index) + 1 if ds.month_index else 0
    K = len(ds.type_names)
    y = np.zeros((T, K), dtype=np.int32)
    for mi, k in zip(ds.month_index, ds.types):
        y[mi, int(k)] = 1
    return y


def prob_from_intensity_integral(integ: np.ndarray) -> np.ndarray:
    # p = 1 - exp(-∫ λ dt)
    return 1.0 - np.exp(-np.clip(integ, 0.0, 1e6))


@dataclass(frozen=True)
class EventPredictionMetrics:
    horizon_months: int
    top1_hit_rate: float | None
    top3_hit_rate: float | None
    top5_hit_rate: float | None
    threshold: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None
    positive_rate: float | None
    predicted_positive_rate: float | None
    always_negative_accuracy: float | None
    always_positive_accuracy: float | None
    balanced_accuracy: float | None
    confusion: dict[str, int] | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "horizon_months": self.horizon_months,
            "top1_hit_rate": self.top1_hit_rate,
            "top3_hit_rate": self.top3_hit_rate,
            "top5_hit_rate": self.top5_hit_rate,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "positive_rate": self.positive_rate,
            "predicted_positive_rate": self.predicted_positive_rate,
            "always_negative_accuracy": self.always_negative_accuracy,
            "always_positive_accuracy": self.always_positive_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "confusion": self.confusion,
        }


def _topk_hit(scores: np.ndarray, label_vec: np.ndarray, k: int) -> int:
    if scores.size == 0:
        return 0
    kk = min(k, scores.size)
    topk = np.argpartition(-scores, kk - 1)[:kk]
    return int(label_vec[topk].max())


def _prf1(pred: np.ndarray, true: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    # pred/true shape: (N,) binary
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    denom_p = tp + fp
    denom_r = tp + fn
    precision = (tp / denom_p) if denom_p > 0 else None
    recall = (tp / denom_r) if denom_r > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision is not None and recall is not None and (precision + recall) > 0) else None
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
    return precision, recall, f1, acc


def _confusion(pred: np.ndarray, true: np.ndarray) -> dict[str, int]:
    # pred/true shape: (N,) binary
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}


def _balanced_accuracy_from_conf(conf: dict[str, int]) -> float | None:
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    tpr_den = tp + fn
    tnr_den = tn + fp
    if tpr_den <= 0 or tnr_den <= 0:
        return None
    tpr = tp / tpr_den
    tnr = tn / tnr_den
    return float(0.5 * (tpr + tnr))


def _logloss(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if scores.size == 0:
        return None
    p = np.clip(scores.astype(np.float64), 1e-9, 1.0 - 1e-9)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if scores.size == 0:
        return None
    p = scores.astype(np.float64)
    y = labels.astype(np.float64)
    return float(np.mean((p - y) ** 2))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    # Same rank-based ROC AUC implementation used elsewhere (no sklearn).
    labels = labels.astype(np.int32)
    if labels.min() == labels.max():
        return None
    order = np.argsort(scores.astype(np.float64))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int(len(labels) - n_pos)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _topk_precision_recall(scores: np.ndarray, label_vec: np.ndarray, k: int) -> tuple[float, float]:
    if scores.size == 0:
        return 0.0, 0.0
    kk = int(min(k, scores.size))
    topk = np.argpartition(-scores, kk - 1)[:kk]
    hits = int(label_vec[topk].sum())
    true_total = int(label_vec.sum())
    prec = hits / kk if kk > 0 else 0.0
    rec = hits / true_total if true_total > 0 else 0.0
    return float(prec), float(rec)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    # Mean of precision at ranks where label=1. Returns None if no positives.
    y = labels.astype(np.int32)
    if y.size == 0:
        return None
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores.astype(np.float64))
    y_sorted = y[order]
    hits = 0
    precs = []
    for i, yi in enumerate(y_sorted, start=1):
        if yi == 1:
            hits += 1
            precs.append(hits / i)
    return float(np.mean(precs)) if precs else None


def _mrr(scores: np.ndarray, labels: np.ndarray) -> float | None:
    # Reciprocal rank of first relevant item. None if no positives.
    y = labels.astype(np.int32)
    if y.size == 0:
        return None
    if int(y.sum()) == 0:
        return None
    order = np.argsort(-scores.astype(np.float64))
    y_sorted = y[order]
    idx = np.where(y_sorted == 1)[0]
    if idx.size == 0:
        return None
    return float(1.0 / (int(idx[0]) + 1))


def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float | None:
    # Binary relevance nDCG@k. Returns None if no positives.
    y = labels.astype(np.int32)
    if y.size == 0:
        return None
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    kk = int(min(k, y.size))
    order = np.argsort(-scores.astype(np.float64))[:kk]
    rel = y[order]
    # DCG with gains rel/log2(i+1)
    denom = np.log2(np.arange(2, kk + 2, dtype=np.float64))
    dcg = float(np.sum(rel.astype(np.float64) / denom))
    # Ideal DCG: all ones in top min(n_pos,kk)
    ideal_rel = np.zeros((kk,), dtype=np.float64)
    ideal_rel[: min(n_pos, kk)] = 1.0
    idcg = float(np.sum(ideal_rel / denom))
    return float(dcg / idcg) if idcg > 0 else None


def _any_event_prob_from_type_probs(p_type: np.ndarray) -> np.ndarray:
    # Using superposition: P(no events) = exp(-∫ sum_k λ_k dt).
    # Given p_k = 1 - exp(-∫ λ_k dt) we get:
    # P(any) = 1 - exp(-∑ -log(1-p_k)) = 1 - ∏ (1-p_k).
    p = np.clip(p_type.astype(np.float64), 0.0, 1.0)
    return 1.0 - np.prod(1.0 - p, axis=1)


def _choose_threshold_on_val_1d(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if scores.size == 0:
        return None
    y = labels.astype(np.int32)
    # If all labels are the same, use median of scores as fallback threshold
    if y.min() == y.max():
        # All positive: use low threshold to predict all positive (matching label)
        if y.min() == 1:
            return float(np.percentile(scores.astype(np.float64), 5))
        # All negative: use high threshold to predict all negative (matching label)
        return float(np.percentile(scores.astype(np.float64), 95))
    cand = np.unique(np.quantile(scores.astype(np.float64), np.linspace(0.01, 0.99, 80)))
    best_t = None
    best_f1 = -1.0
    for t in cand:
        pred = (scores >= t).astype(np.int32)
        _, _, f1, _ = _prf1(pred, y)
        if f1 is None:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


@dataclass(frozen=True)
class MonthlyRealisticEval:
    horizon_months: int
    any_event_auc: float | None
    any_event_brier: float | None
    any_event_logloss: float | None
    threshold: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None
    balanced_accuracy: float | None
    positive_rate: float | None  # fraction of months with any-event label=1
    predicted_positive_rate: float | None  # fraction of months predicted as positive under threshold
    confusion: dict[str, int] | None
    topk_hit_rate: dict[str, float] | None  # per-month: any true type in top-k
    precision_at_k: dict[str, float] | None  # per-month avg precision@k over types
    recall_at_k: dict[str, float] | None  # per-month avg recall@k over types (0 if no events)
    avg_true_types_per_month: float | None
    map: float | None  # mean average precision across months (only months with any positives)
    mrr: float | None  # mean reciprocal rank across months (only months with any positives)
    ndcg_at_k: dict[str, float] | None  # nDCG@k across months (only months with any positives)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "horizon_months": self.horizon_months,
            "any_event_auc": self.any_event_auc,
            "any_event_brier": self.any_event_brier,
            "any_event_logloss": self.any_event_logloss,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "positive_rate": self.positive_rate,
            "predicted_positive_rate": self.predicted_positive_rate,
            "confusion": self.confusion,
            "topk_hit_rate": self.topk_hit_rate,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "avg_true_types_per_month": self.avg_true_types_per_month,
            "map": self.map,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
        }


def eval_monthly_realistic(
    *,
    model,
    ds: EventDataset,
    split_for_threshold: str,
    split_for_scoring: str,
    horizon_months: int,
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> MonthlyRealisticEval:
    # Use the same no-lookahead monthly probs/labels as the tearsheet, but score in a realistic way:
    # - month-level "any event occurs" (single binary decision per month)
    # - month-level "top-k type ranking" (hit rate, precision@k, recall@k)
    probs_val, labels_val = compute_probs_labels_no_lookahead(
        model=model, ds=ds, split=split_for_threshold, horizon_months=horizon_months
    )
    probs_te, labels_te = compute_probs_labels_no_lookahead(
        model=model, ds=ds, split=split_for_scoring, horizon_months=horizon_months
    )
    if probs_te.size == 0:
        return MonthlyRealisticEval(
            horizon_months=horizon_months,
            any_event_auc=None,
            any_event_brier=None,
            any_event_logloss=None,
            threshold=None,
            precision=None,
            recall=None,
            f1=None,
            accuracy=None,
            balanced_accuracy=None,
            positive_rate=None,
            predicted_positive_rate=None,
            confusion=None,
            topk_hit_rate=None,
            precision_at_k=None,
            recall_at_k=None,
            avg_true_types_per_month=None,
        )

    # Month-level labels: any type happens within horizon.
    y_val_any = (labels_val.max(axis=1) > 0).astype(np.int32) if labels_val.size else np.zeros((0,), dtype=np.int32)
    y_te_any = (labels_te.max(axis=1) > 0).astype(np.int32)
    p_val_any = _any_event_prob_from_type_probs(probs_val) if probs_val.size else np.zeros((0,), dtype=np.float64)
    p_te_any = _any_event_prob_from_type_probs(probs_te)

    any_auc = _auc(p_te_any, y_te_any)
    any_brier = _brier(p_te_any, y_te_any)
    any_logloss = _logloss(p_te_any, y_te_any)

    threshold = _choose_threshold_on_val_1d(p_val_any, y_val_any) if p_val_any.size else None
    precision = recall = f1 = acc = bal_acc = None
    conf = None
    pos_rate = float(y_te_any.mean()) if y_te_any.size else None
    pred_rate = None
    if threshold is not None:
        pred_any = (p_te_any >= float(threshold)).astype(np.int32)
        precision, recall, f1, acc = _prf1(pred_any, y_te_any)
        conf = _confusion(pred_any, y_te_any)
        bal_acc = _balanced_accuracy_from_conf(conf)
        pred_rate = float(pred_any.mean()) if pred_any.size else None

    # Month-level ranking metrics over types.
    hit_rates: dict[str, float] = {}
    prec_at_k: dict[str, float] = {}
    rec_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}
    true_types_per_month = labels_te.sum(axis=1).astype(np.float64)
    avg_true_types = float(true_types_per_month.mean()) if true_types_per_month.size else None

    aps: list[float] = []
    rrs: list[float] = []
    for k in ks:
        kk = int(min(k, probs_te.shape[1]))
        hits = []
        ps = []
        rs = []
        nds = []
        for p_row, y_row in zip(probs_te, labels_te):
            hits.append(int(y_row[np.argpartition(-p_row, kk - 1)[:kk]].max()) if kk > 0 else 0)
            p_k, r_k = _topk_precision_recall(p_row, y_row, kk if kk > 0 else 1)
            ps.append(p_k)
            rs.append(r_k)
            nd = _ndcg_at_k(p_row, y_row, kk if kk > 0 else 1)
            if nd is not None:
                nds.append(nd)
        hit_rates[str(k)] = float(np.mean(hits)) if hits else float("nan")
        prec_at_k[str(k)] = float(np.mean(ps)) if ps else float("nan")
        rec_at_k[str(k)] = float(np.mean(rs)) if rs else float("nan")
        ndcg_at_k[str(k)] = float(np.mean(nds)) if nds else float("nan")

    for p_row, y_row in zip(probs_te, labels_te):
        ap = _average_precision(p_row, y_row)
        if ap is not None:
            aps.append(ap)
        rr = _mrr(p_row, y_row)
        if rr is not None:
            rrs.append(rr)

    return MonthlyRealisticEval(
        horizon_months=horizon_months,
        any_event_auc=any_auc,
        any_event_brier=any_brier,
        any_event_logloss=any_logloss,
        threshold=float(threshold) if threshold is not None else None,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        positive_rate=pos_rate,
        predicted_positive_rate=pred_rate,
        confusion=conf,
        topk_hit_rate=hit_rates,
        precision_at_k=prec_at_k,
        recall_at_k=rec_at_k,
        avg_true_types_per_month=avg_true_types,
        map=float(np.mean(aps)) if aps else None,
        mrr=float(np.mean(rrs)) if rrs else None,
        ndcg_at_k=ndcg_at_k,
    )

def _month_range_from_split(ds: EventDataset, split: str) -> tuple[int, int] | None:
    ev_mask = np.array(ds.split_mask[split], dtype=bool)
    months_in_split = np.array(ds.month_index, dtype=int)[ev_mask]
    if months_in_split.size == 0:
        return None
    return int(months_in_split.min()), int(months_in_split.max())


def _integrate_intensity_no_lookahead(
    *,
    model,
    ds: EventDataset,
    t0: float,
    t1: float,
    grid_points_per_month: int,
) -> np.ndarray:
    """Compute per-type integral of λ_k over [t0, t1] using ONLY history < t0."""
    device = next(model.parameters()).device

    # History = all events strictly before t0
    times_all = np.array(ds.times, dtype=np.float32)
    hist_mask = times_all < float(t0)
    times = torch.tensor(times_all[hist_mask], dtype=torch.float32, device=device)
    types = torch.tensor(np.array(ds.types, dtype=np.int64)[hist_mask], dtype=torch.long, device=device)
    marks = torch.tensor(np.array(ds.marks, dtype=np.float32)[hist_mask], dtype=torch.float32, device=device)

    grid = torch.linspace(
        float(t0),
        float(t1),
        steps=max(2, int((t1 - t0) * grid_points_per_month) + 1),
        device=device,
    )

    with torch.no_grad():
        lam = model.forward_intensity_path(times, types, marks, query_times=grid)  # (G,K)
        integ = torch.trapezoid(lam, grid, dim=0)  # (K,)
    return integ.detach().cpu().numpy()


def choose_threshold_on_val(
    *,
    probs: np.ndarray,   # (N, K)
    labels: np.ndarray,  # (N, K)
) -> float | None:
    # Global threshold chosen to maximize macro F1 on the validation set.
    if probs.size == 0:
        return None
    flat_p = probs.reshape(-1)
    flat_y = labels.reshape(-1).astype(np.int32)
    if flat_y.min() == flat_y.max():
        return None

    # Candidate thresholds = quantiles of predicted probs (robust grid).
    qs = np.linspace(0.5, 0.99, 40)
    cand = np.unique(np.quantile(flat_p, qs))
    best_t = None
    best_f1 = -1.0
    for t in cand:
        pred = (flat_p >= t).astype(np.int32)
        p, r, f1, _ = _prf1(pred, flat_y)
        if f1 is None:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def eval_event_prediction(
    *,
    model,
    ds: EventDataset,
    split_for_threshold: str,
    split_for_scoring: str,
    horizon_months: int,
    grid_points_per_month: int = 4,
) -> EventPredictionMetrics:
    y = month_event_matrix(ds)  # (T,K)
    T, K = y.shape

    # Compute per-month per-type probabilities for the threshold-selection split (val).
    def compute_probs_and_labels(split: str) -> tuple[np.ndarray, np.ndarray]:
        rng = _month_range_from_split(ds, split)
        if rng is None:
            return np.zeros((0, K), dtype=np.float64), np.zeros((0, K), dtype=np.int32)
        m_start, m_end = rng

        probs_list = []
        labels_list = []
        for m in range(m_start, m_end + 1):
            t0 = float(m)
            t1 = float(min(m + horizon_months, T))
            if t1 <= t0:
                continue
            integ = _integrate_intensity_no_lookahead(
                model=model,
                ds=ds,
                t0=t0,
                t1=t1,
                grid_points_per_month=grid_points_per_month,
            )
            prob = prob_from_intensity_integral(integ)  # (K,)

            end_m = min(m + horizon_months, T - 1)
            future = y[m + 1 : end_m + 1]
            lab = future.max(axis=0) if future.size else np.zeros((K,), dtype=np.int32)

            probs_list.append(prob.astype(np.float64))
            labels_list.append(lab.astype(np.int32))

        if not probs_list:
            return np.zeros((0, K), dtype=np.float64), np.zeros((0, K), dtype=np.int32)
        return np.stack(probs_list, axis=0), np.stack(labels_list, axis=0)

    probs_val, labels_val = compute_probs_and_labels(split_for_threshold)
    threshold = choose_threshold_on_val(probs=probs_val, labels=labels_val)

    probs_te, labels_te = compute_probs_and_labels(split_for_scoring)
    if probs_te.size == 0:
        return EventPredictionMetrics(
            horizon_months=horizon_months,
            top1_hit_rate=None,
            top3_hit_rate=None,
            top5_hit_rate=None,
            threshold=threshold,
            precision=None,
            recall=None,
            f1=None,
            accuracy=None,
            positive_rate=None,
            predicted_positive_rate=None,
            always_negative_accuracy=None,
            always_positive_accuracy=None,
            balanced_accuracy=None,
            confusion=None,
        )

    # Top-k hit rates: per month, did any of the top-k predicted event types happen?
    top1 = np.mean([_topk_hit(p, yv, 1) for p, yv in zip(probs_te, labels_te)])
    top3 = np.mean([_topk_hit(p, yv, 3) for p, yv in zip(probs_te, labels_te)])
    top5 = np.mean([_topk_hit(p, yv, 5) for p, yv in zip(probs_te, labels_te)])

    # Threshold metrics over all (month,type) pairs.
    if threshold is None:
        return EventPredictionMetrics(
            horizon_months=horizon_months,
            top1_hit_rate=float(top1),
            top3_hit_rate=float(top3),
            top5_hit_rate=float(top5),
            threshold=None,
            precision=None,
            recall=None,
            f1=None,
            accuracy=None,
            positive_rate=None,
            predicted_positive_rate=None,
            always_negative_accuracy=None,
            always_positive_accuracy=None,
            balanced_accuracy=None,
            confusion=None,
        )

    pred = (probs_te.reshape(-1) >= threshold).astype(np.int32)
    true = labels_te.reshape(-1).astype(np.int32)
    precision, recall, f1, acc = _prf1(pred, true)
    conf = _confusion(pred, true)
    pos_rate = float(true.mean()) if true.size else None
    pred_rate = float(pred.mean()) if pred.size else None
    always_neg_acc = float((true == 0).mean()) if true.size else None
    always_pos_acc = float((true == 1).mean()) if true.size else None
    bal_acc = _balanced_accuracy_from_conf(conf)

    return EventPredictionMetrics(
        horizon_months=horizon_months,
        top1_hit_rate=float(top1),
        top3_hit_rate=float(top3),
        top5_hit_rate=float(top5),
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=acc,
        positive_rate=pos_rate,
        predicted_positive_rate=pred_rate,
        always_negative_accuracy=always_neg_acc,
        always_positive_accuracy=always_pos_acc,
        balanced_accuracy=bal_acc,
        confusion=conf,
    )


def compute_probs_labels_no_lookahead(
    *,
    model,
    ds: EventDataset,
    split: str,
    horizon_months: int,
    grid_points_per_month: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) for each month in split, no-lookahead.

    probs: (M, K) where probs[m,k] = P(type k occurs within next H months | history < month_m)
    labels: (M, K) binary (whether type k occurs within next H months)
    """
    y = month_event_matrix(ds)
    T, K = y.shape
    rng = _month_range_from_split(ds, split)
    if rng is None:
        return np.zeros((0, K), dtype=np.float64), np.zeros((0, K), dtype=np.int32)
    m_start, m_end = rng

    probs_list = []
    labels_list = []
    for m in range(m_start, m_end + 1):
        t0 = float(m)
        t1 = float(min(m + horizon_months, T))
        if t1 <= t0:
            continue
        integ = _integrate_intensity_no_lookahead(
            model=model,
            ds=ds,
            t0=t0,
            t1=t1,
            grid_points_per_month=grid_points_per_month,
        )
        prob = prob_from_intensity_integral(integ)

        end_m = min(m + horizon_months, T - 1)
        future = y[m + 1 : end_m + 1]
        lab = future.max(axis=0) if future.size else np.zeros((K,), dtype=np.int32)

        probs_list.append(prob.astype(np.float64))
        labels_list.append(lab.astype(np.int32))

    if not probs_list:
        return np.zeros((0, K), dtype=np.float64), np.zeros((0, K), dtype=np.int32)
    return np.stack(probs_list, axis=0), np.stack(labels_list, axis=0)


