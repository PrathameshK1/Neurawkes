from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import base64
import html

from nhp_torch.baselines.binned import (
    PoissonIID,
    build_binned_counts,
    fit_hawkes_ar1_poisson,
    hawkes_ar1_nll_from_state,
    split_indices,
)
from nhp_torch.baselines.shock_classifier import fit_logistic_multilabel
from nhp_torch.baselines.var import eval_var1_nextmonth_auc
from nhp_torch.config.io import Config
from nhp_torch.data.events import EventDataset
from nhp_torch.eval.event_prediction import compute_probs_labels_no_lookahead, eval_monthly_realistic
from nhp_torch.eval.metrics import evaluate_checkpoint
from nhp_torch.models.nhp import NeuralHawkes
from nhp_torch.utils import write_json


def _safe_mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _prf1(pred: np.ndarray, true: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    denom_p = tp + fp
    denom_r = tp + fn
    precision = (tp / denom_p) if denom_p > 0 else None
    recall = (tp / denom_r) if denom_r > 0 else None
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
    return precision, recall, f1, acc


def _confusion(pred: np.ndarray, true: np.ndarray) -> dict[str, int]:
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


def _plot_topk_curve(out_path: Path, probs: np.ndarray, labels: np.ndarray, ks=(1, 3, 5, 10, 20)) -> dict[str, float]:
    hit_rates = {}
    for k in ks:
        kk = min(k, probs.shape[1])
        hits = []
        for p, y in zip(probs, labels):
            topk = np.argpartition(-p, kk - 1)[:kk]
            hits.append(int(y[topk].max()))
        hit_rates[str(k)] = float(np.mean(hits)) if hits else float("nan")

    plt.figure(figsize=(6, 4))
    plt.plot([int(k) for k in hit_rates.keys()], list(hit_rates.values()), marker="o")
    plt.xlabel("k (predict top-k event types)")
    plt.ylabel("hit rate (any true event in top-k)")
    plt.title("Top-k hit-rate on test")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return hit_rates


def _plot_prob_hist(out_path: Path, probs: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(probs.reshape(-1), bins=50, alpha=0.9)
    plt.xlabel("predicted probability")
    plt.ylabel("count")
    plt.title("Distribution of predicted probabilities (test, all types/months)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_reliability(out_path: Path, probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict[str, float]:
    # Global reliability over all (month,type) pairs.
    p = probs.reshape(-1)
    y = labels.reshape(-1).astype(np.int32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(p, edges) - 1
    xs = []
    ys = []
    counts = []
    for b in range(bins):
        m = bin_ids == b
        if not m.any():
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        counts.append(int(m.sum()))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(xs, ys, marker="o")
    plt.xlabel("predicted probability")
    plt.ylabel("empirical frequency")
    plt.title("Reliability (global, test)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

    # Brier score (global)
    brier = float(np.mean((p - y) ** 2)) if p.size else float("nan")
    return {"brier": brier, "bins_used": len(xs), "pairs": int(p.size)}


def _load_model(cfg: Config, ds: EventDataset, checkpoint_path: Path) -> NeuralHawkes:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    K = len(ds.type_names)
    model = NeuralHawkes(
        num_types=K,
        embed_dim=int(cfg.model["embed_dim"]),
        hidden_dim=int(cfg.model["hidden_dim"]),
        use_marks=bool(cfg.model["use_marks"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_tearsheet(*, cfg: Config, ds: EventDataset, checkpoint_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Core metrics (LL, AUC, baseline comparisons etc.)
    metrics = evaluate_checkpoint(cfg, ds, checkpoint_path=checkpoint_path)

    # 2) Explicit “did it happen?” evaluation with no-lookahead probabilities
    model = _load_model(cfg, ds, checkpoint_path)
    horizons = list(cfg.eval.get("horizons_months", [1, 3]))
    tearsheet_extra: dict[str, Any] = {"event_prediction": {}}
    tearsheet_extra["event_prediction_monthly"] = {}

    for h in horizons:
        probs, labels = compute_probs_labels_no_lookahead(model=model, ds=ds, split="test", horizon_months=int(h))
        if probs.size == 0:
            continue

        # Realistic month-level metrics (any-event + top-k type ranking), threshold tuned on val.
        monthly = eval_monthly_realistic(
            model=model,
            ds=ds,
            split_for_threshold="val",
            split_for_scoring="test",
            horizon_months=int(h),
        )
        tearsheet_extra["event_prediction_monthly"][str(h)] = monthly.to_jsonable()

        # Threshold chosen on val (global) for interpretability (precision/recall tradeoff).
        probs_val, labels_val = compute_probs_labels_no_lookahead(model=model, ds=ds, split="val", horizon_months=int(h))
        threshold = None
        if probs_val.size:
            cand = np.unique(np.quantile(probs_val.reshape(-1), np.linspace(0.5, 0.99, 40)))
            best_t = None
            best_f1 = -1.0
            for t in cand:
                pred = (probs_val.reshape(-1) >= t).astype(np.int32)
                true = labels_val.reshape(-1).astype(np.int32)
                _, _, f1, _ = _prf1(pred, true)
                if f1 is not None and f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            threshold = best_t

        # Compute threshold metrics on test
        precision = recall = f1 = acc = None
        base_rate = pred_rate = always_neg_acc = always_pos_acc = bal_acc = None
        conf: dict[str, int] | None = None
        if threshold is not None:
            pred = (probs.reshape(-1) >= threshold).astype(np.int32)
            true = labels.reshape(-1).astype(np.int32)
            precision, recall, f1, acc = _prf1(pred, true)
            conf = _confusion(pred, true)
            base_rate = float(true.mean()) if true.size else None
            pred_rate = float(pred.mean()) if pred.size else None
            always_neg_acc = float((true == 0).mean()) if true.size else None
            always_pos_acc = float((true == 1).mean()) if true.size else None
            bal_acc = _balanced_accuracy_from_conf(conf)

        # Per-type F1 leaderboard (only types with any positives)
        per_type_f1: dict[str, float] = {}
        if threshold is not None:
            pred_m = (probs >= threshold).astype(np.int32)
            for k, name in enumerate(ds.type_names):
                yy = labels[:, k]
                if yy.min() == yy.max():
                    continue
                p, r, ff, _ = _prf1(pred_m[:, k], yy)
                if ff is not None:
                    per_type_f1[name] = float(ff)

        # Plots
        hit_path = out_dir / f"topk_hit_h{h}.png"
        hist_path = out_dir / f"prob_hist_h{h}.png"
        rel_path = out_dir / f"reliability_h{h}.png"
        hit_rates = _plot_topk_curve(hit_path, probs, labels)
        _plot_prob_hist(hist_path, probs)
        reliability = _plot_reliability(rel_path, probs, labels)

        tearsheet_extra["event_prediction"][str(h)] = {
            "horizon_months": int(h),
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
            "positive_rate": base_rate,
            "predicted_positive_rate": pred_rate,
            "always_negative_accuracy": always_neg_acc,
            "always_positive_accuracy": always_pos_acc,
            "balanced_accuracy": bal_acc,
            "confusion": conf,
            "topk_hit_rate": hit_rates,
            "reliability": reliability,
            "top_f1_types": dict(sorted(per_type_f1.items(), key=lambda kv: -kv[1])[:15]),
            "plots": {
                "topk_hit_curve": str(hit_path),
                "prob_hist": str(hist_path),
                "reliability": str(rel_path),
            },
        }

    # 3) Baselines (recomputed for this tearsheet for completeness)
    binned = build_binned_counts(cfg, ds)
    idxs = split_indices(cfg, binned)
    counts_tr = binned.counts[idxs["train"]]
    counts_val = binned.counts[idxs["val"]]
    counts_te = binned.counts[idxs["test"]]
    poi = PoissonIID.fit(counts_tr)
    hawkes_ar1 = fit_hawkes_ar1_poisson(counts=binned.counts, idx_train=idxs["train"], idx_val=idxs["val"])
    baselines = {
        "poisson_iid": {
            "train_nll": poi.nll(counts_tr),
            "val_nll": poi.nll(counts_val),
            "test_nll": poi.nll(counts_te),
        },
        "hawkes_ar1_poisson": {
            "train_nll": hawkes_ar1_nll_from_state(counts=counts_tr, state=hawkes_ar1.model_state),
            "val_nll": hawkes_ar1_nll_from_state(counts=counts_val, state=hawkes_ar1.model_state),
            "test_nll": hawkes_ar1_nll_from_state(counts=counts_te, state=hawkes_ar1.model_state),
        },
        "logistic_multilabel": {"test_macro_auc": fit_logistic_multilabel(cfg=cfg, ds=ds, hidden=None)[1].macro_auc},
        "mlp_multilabel": {"test_macro_auc": fit_logistic_multilabel(cfg=cfg, ds=ds, hidden=64)[1].macro_auc},
        "var1_threshold_proxy": {"test_macro_auc": eval_var1_nextmonth_auc(cfg=cfg, ds=ds)},
    }

    # 4) Write artifacts
    out_json = {
        "metrics": metrics,
        "tearsheet": tearsheet_extra,
        "baselines_recomputed": baselines,
    }
    write_json(out_dir / "tearsheet.json", out_json)

    # 5) Markdown report (QuantStats-like “tear sheet”)
    md = []
    md.append("## Model evaluation tearsheet (event prediction)\n\n")
    md.append(f"- checkpoint: `{checkpoint_path}`\n")
    md.append(f"- train_ll: {metrics.get('train_ll')}\n")
    md.append(f"- val_ll: {metrics.get('val_ll')} (best_val_ll: {metrics.get('best_val_ll')})\n")
    md.append(f"- test_ll: {metrics.get('test_ll')}\n")
    md.append("\n### Core interpretation\n")
    md.append("- `test_ll` is a (Monte Carlo) estimate of out-of-sample point-process log-likelihood; higher is better.\n")
    md.append("- `event_prediction` below turns intensities into explicit “did an event occur within H months?” predictions.\n")
    md.append("- For rare events, **F1 + top-k hit rate** are usually more meaningful than raw accuracy.\n\n")

    md.append("## Event prediction (no-lookahead)\n\n")
    for h in horizons:
        obj_m = tearsheet_extra["event_prediction_monthly"].get(str(h))
        if not obj_m:
            continue
        md.append(f"### Horizon H={h} months\n\n")
        md.append("Month-level (realistic) any-event decision:\n")
        md.append(f"- any_event_auc: {obj_m.get('any_event_auc')}\n")
        md.append(f"- any_event_brier: {obj_m.get('any_event_brier')}\n")
        md.append(f"- any_event_logloss: {obj_m.get('any_event_logloss')}\n")
        md.append(f"- threshold (val-tuned, maximize F1): {obj_m.get('threshold')}\n")
        md.append(f"- positive_rate (months): {obj_m.get('positive_rate')}\n")
        md.append(f"- predicted_positive_rate (months): {obj_m.get('predicted_positive_rate')}\n")
        md.append(f"- balanced_accuracy: {obj_m.get('balanced_accuracy')}\n")
        md.append(f"- confusion: {obj_m.get('confusion')}\n")
        md.append(
            f"- precision / recall / f1 / accuracy: {obj_m.get('precision')} / {obj_m.get('recall')} / {obj_m.get('f1')} / {obj_m.get('accuracy')}\n"
        )
        md.append("\nType ranking (realistic):\n")
        md.append(f"- avg_true_types_per_month: {obj_m.get('avg_true_types_per_month')}\n")
        md.append(f"- MAP: {obj_m.get('map')}\n")
        md.append(f"- MRR: {obj_m.get('mrr')}\n")
        md.append(f"- top-k hit rates: {obj_m.get('topk_hit_rate')}\n")
        md.append(f"- precision@k: {obj_m.get('precision_at_k')}\n")
        md.append(f"- recall@k: {obj_m.get('recall_at_k')}\n\n")
        md.append(f"- nDCG@k: {obj_m.get('ndcg_at_k')}\n\n")

        # Keep the legacy pairwise (month×type) thresholded metrics for debugging only.
        obj = tearsheet_extra["event_prediction"].get(str(h)) or {}
        md.append("Plots:\n")
        md.append(f"- `{obj['plots']['topk_hit_curve']}`\n")
        md.append(f"- `{obj['plots']['prob_hist']}`\n")
        md.append(f"- `{obj['plots']['reliability']}`\n\n")
        md.append("Top per-type F1 (thresholded):\n")
        for name, val in (obj.get("top_f1_types") or {}).items():
            md.append(f"- {name}: {val:.3f}\n")
        md.append("\n")

    md.append("## Baselines (recomputed)\n\n")
    md.append(f"- poisson_iid test_nll: {baselines['poisson_iid']['test_nll']:.3f}\n")
    md.append(f"- hawkes_ar1_poisson test_nll: {baselines['hawkes_ar1_poisson']['test_nll']:.3f}\n")
    md.append(f"- logistic_multilabel test_macro_auc: {baselines['logistic_multilabel']['test_macro_auc']}\n")
    md.append(f"- mlp_multilabel test_macro_auc: {baselines['mlp_multilabel']['test_macro_auc']}\n")
    md.append(f"- var1_threshold_proxy test_macro_auc: {baselines['var1_threshold_proxy']['test_macro_auc']}\n\n")

    md.append("## Conclusion (what it means)\n\n")
    md.append(
        "On this dataset, the Neural Hawkes model is learning a coherent intensity process (train/val LL improves), "
        "but the practical event-detection quality on 2022–2023 depends strongly on horizon and the decision rule. "
        "If horizon AUC and top-k hit rates are near random, the model’s intensities are not ranking future shocks well. "
        "When simple baselines (e.g., VAR(1) on z-changes) outperform the Hawkes intensity ranking, it suggests the signal here "
        "is more about discrete-time autocorrelation/regime effects than self-exciting event contagion at this monthly resolution.\n"
    )
    md.append(
        "Recommended next steps for research quality: (1) tune event thresholds to stabilize event frequency, "
        "(2) evaluate with a walk-forward protocol, and (3) restrict to a smaller, higher-signal subset of columns "
        "(e.g., energy + a few food items) to reduce the 66-type sparsity problem.\n"
    )

    (out_dir / "tearsheet.md").write_text("".join(md), encoding="utf-8")

    # 6) Self-contained HTML (QuantStats-like): embed plots directly in the HTML as data URIs.
    def img_data_uri(path: str | Path) -> str | None:
        p = Path(path)
        if not p.exists():
            return None
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    html_lines: list[str] = []
    html_lines.append("<!doctype html><html><head><meta charset='utf-8'/>")
    html_lines.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    html_lines.append("<title>Neural Hawkes Tearsheets</title>")
    html_lines.append(
        "<style>"
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;max-width:1100px}"
        "h1,h2,h3{margin:18px 0 8px}"
        "code,pre{background:#f6f8fa;padding:2px 6px;border-radius:4px}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
        ".card{border:1px solid #e5e7eb;border-radius:8px;padding:14px}"
        ".muted{color:#6b7280}"
        "img{max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:6px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #e5e7eb;padding:6px 8px;font-size:13px}"
        "</style>"
    )
    html_lines.append("</head><body>")

    html_lines.append("<h1>Model evaluation tearsheet (event prediction)</h1>")
    html_lines.append("<div class='card'>")
    html_lines.append(f"<div><span class='muted'>checkpoint</span>: <code>{esc(str(checkpoint_path))}</code></div>")
    html_lines.append(f"<div><span class='muted'>train_ll</span>: {esc(str(metrics.get('train_ll')))}</div>")
    html_lines.append(
        f"<div><span class='muted'>val_ll</span>: {esc(str(metrics.get('val_ll')))} "
        f"(best_val_ll: {esc(str(metrics.get('best_val_ll')))})</div>"
    )
    html_lines.append(f"<div><span class='muted'>test_ll</span>: {esc(str(metrics.get('test_ll')))}</div>")
    html_lines.append("</div>")

    html_lines.append("<h2>Interpretation</h2>")
    html_lines.append("<div class='card'>")
    html_lines.append("<ul>")
    html_lines.append("<li><code>test_ll</code> is an out-of-sample point-process log-likelihood estimate; higher is better.</li>")
    html_lines.append("<li>For rare events, focus on <b>F1</b> and <b>top-k hit rate</b>, not raw accuracy.</li>")
    html_lines.append("<li>Reliability plots show whether predicted probabilities are calibrated.</li>")
    html_lines.append("</ul>")
    html_lines.append("</div>")

    html_lines.append("<h2>Event prediction (no-lookahead)</h2>")
    for h in horizons:
        obj_m = tearsheet_extra["event_prediction_monthly"].get(str(h))
        obj = tearsheet_extra["event_prediction"].get(str(h)) or {}
        if not obj_m:
            continue
        html_lines.append(f"<h3>Horizon H={int(h)} months</h3>")
        html_lines.append("<div class='card'>")
        html_lines.append("<div class='muted' style='margin-bottom:8px'><b>Month-level (realistic) any-event decision</b></div>")
        html_lines.append("<table>")
        html_lines.append(
            "<tr>"
            "<th>any_event_auc</th>"
            "<th>any_event_brier</th>"
            "<th>any_event_logloss</th>"
            "<th>threshold (val-tuned)</th>"
            "<th>positive_rate (months)</th>"
            "<th>predicted_positive_rate</th>"
            "<th>balanced_acc</th>"
            "<th>precision</th>"
            "<th>recall</th>"
            "<th>F1</th>"
            "<th>accuracy</th>"
            "</tr>"
        )
        html_lines.append(
            "<tr>"
            f"<td>{esc(str(obj_m.get('any_event_auc')))}</td>"
            f"<td>{esc(str(obj_m.get('any_event_brier')))}</td>"
            f"<td>{esc(str(obj_m.get('any_event_logloss')))}</td>"
            f"<td>{esc(str(obj_m.get('threshold')))}</td>"
            f"<td>{esc(str(obj_m.get('positive_rate')))}</td>"
            f"<td>{esc(str(obj_m.get('predicted_positive_rate')))}</td>"
            f"<td>{esc(str(obj_m.get('balanced_accuracy')))}</td>"
            f"<td>{esc(str(obj_m.get('precision')))}</td>"
            f"<td>{esc(str(obj_m.get('recall')))}</td>"
            f"<td>{esc(str(obj_m.get('f1')))}</td>"
            f"<td>{esc(str(obj_m.get('accuracy')))}</td>"
            "</tr>"
        )
        html_lines.append("</table>")
        html_lines.append(f"<div class='muted' style='margin-top:8px'>confusion: {esc(str(obj_m.get('confusion')))}</div>")
        html_lines.append("<div class='muted' style='margin-top:10px'><b>Type ranking (realistic)</b></div>")
        html_lines.append(f"<div class='muted'>avg_true_types_per_month: {esc(str(obj_m.get('avg_true_types_per_month')))}</div>")
        html_lines.append(f"<div class='muted'>MAP: {esc(str(obj_m.get('map')))}</div>")
        html_lines.append(f"<div class='muted'>MRR: {esc(str(obj_m.get('mrr')))}</div>")
        html_lines.append(f"<div class='muted'>top-k hit rates: {esc(str(obj_m.get('topk_hit_rate')))}</div>")
        html_lines.append(f"<div class='muted'>precision@k: {esc(str(obj_m.get('precision_at_k')))}</div>")
        html_lines.append(f"<div class='muted'>recall@k: {esc(str(obj_m.get('recall_at_k')))}</div>")
        html_lines.append(f"<div class='muted'>nDCG@k: {esc(str(obj_m.get('ndcg_at_k')))}</div>")
        html_lines.append("<div class='muted' style='margin-top:10px'><b>Debug (legacy month×type thresholding)</b></div>")
        html_lines.append(f"<div class='muted'>top-k hit rates (same as above plot): {esc(str(obj.get('topk_hit_rate')))}</div>")
        html_lines.append(f"<div class='muted'>reliability (pairwise): {esc(str(obj.get('reliability')))}</div>")
        html_lines.append("</div>")

        img1 = img_data_uri(obj["plots"]["topk_hit_curve"])
        img2 = img_data_uri(obj["plots"]["prob_hist"])
        img3 = img_data_uri(obj["plots"]["reliability"])
        html_lines.append("<div class='grid'>")
        if img1:
            html_lines.append(f"<div class='card'><div class='muted'>Top-k hit curve</div><img src='{img1}'/></div>")
        if img2:
            html_lines.append(f"<div class='card'><div class='muted'>Probability histogram</div><img src='{img2}'/></div>")
        html_lines.append("</div>")
        if img3:
            html_lines.append(f"<div class='card'><div class='muted'>Reliability</div><img src='{img3}'/></div>")

        top_f1 = obj.get("top_f1_types") or {}
        if top_f1:
            html_lines.append("<div class='card'>")
            html_lines.append("<div class='muted'>Top per-type F1 (thresholded)</div>")
            html_lines.append("<table>")
            html_lines.append("<tr><th>event_type</th><th>F1</th></tr>")
            for name, val in top_f1.items():
                html_lines.append(f"<tr><td>{esc(name)}</td><td>{esc(f'{val:.3f}')}</td></tr>")
            html_lines.append("</table>")
            html_lines.append("</div>")

    html_lines.append("<h2>Baselines</h2>")
    html_lines.append("<div class='card'>")
    html_lines.append("<table>")
    html_lines.append("<tr><th>baseline</th><th>metric</th><th>value</th></tr>")
    html_lines.append(f"<tr><td>poisson_iid</td><td>test_nll</td><td>{esc(str(baselines['poisson_iid']['test_nll']))}</td></tr>")
    html_lines.append(f"<tr><td>hawkes_ar1_poisson</td><td>test_nll</td><td>{esc(str(baselines['hawkes_ar1_poisson']['test_nll']))}</td></tr>")
    html_lines.append(f"<tr><td>logistic_multilabel</td><td>test_macro_auc</td><td>{esc(str(baselines['logistic_multilabel']['test_macro_auc']))}</td></tr>")
    html_lines.append(f"<tr><td>mlp_multilabel</td><td>test_macro_auc</td><td>{esc(str(baselines['mlp_multilabel']['test_macro_auc']))}</td></tr>")
    html_lines.append(f"<tr><td>var1_threshold_proxy</td><td>test_macro_auc</td><td>{esc(str(baselines['var1_threshold_proxy']['test_macro_auc']))}</td></tr>")
    html_lines.append("</table>")
    html_lines.append("</div>")

    html_lines.append("<h2>Conclusion</h2>")
    html_lines.append("<div class='card'>")
    html_lines.append(
        "<p>"
        "This tearsheet summarizes whether the model can <b>rank and detect</b> future shock-events out-of-sample (2022–2023). "
        "If top-k hit rates and F1 are low, the model is not reliably predicting rare events at this horizon and thresholding regime. "
        "When simpler discrete-time baselines outperform, it indicates the dominant signal is likely autocorrelation/regime behavior rather "
        "than self-excitation captured by Hawkes at monthly resolution."
        "</p>"
    )
    html_lines.append(
        "<p class='muted'>Next steps: tune the event threshold, reduce the event-type universe, and use walk-forward evaluation.</p>"
    )
    html_lines.append("</div>")

    html_lines.append("</body></html>")
    (out_dir / "tearsheet.html").write_text("".join(html_lines), encoding="utf-8")


