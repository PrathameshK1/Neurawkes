from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nhp_torch.config.io import load_config
from nhp_torch.baselines.binned import PoissonIID, build_binned_counts, fit_hawkes_ar1_poisson, hawkes_ar1_nll_from_state, split_indices
from nhp_torch.baselines.shock_classifier import fit_logistic_multilabel
from nhp_torch.baselines.var import eval_var1_nextmonth_auc
from nhp_torch.data.events import build_event_dataset
from nhp_torch.eval.metrics import evaluate_checkpoint
from nhp_torch.eval.plots import save_bar_top_types
from nhp_torch.utils import ensure_dir, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained Neural Hawkes checkpoint.")
    ap.add_argument(
        "--config",
        default="nhp_torch/config/default.toml",
        help="Path to TOML config.",
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved checkpoint .pt file.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = build_event_dataset(cfg)

    artifacts_dir = ensure_dir(cfg.paths["artifacts_dir"])
    out_dir = ensure_dir(artifacts_dir / "eval")

    metrics = evaluate_checkpoint(cfg, ds, checkpoint_path=Path(args.checkpoint))

    # Baselines (binned / discrete-time)
    binned = build_binned_counts(cfg, ds)
    idxs = split_indices(cfg, binned)
    counts_tr = binned.counts[idxs["train"]]
    counts_val = binned.counts[idxs["val"]]
    counts_te = binned.counts[idxs["test"]]

    poi = PoissonIID.fit(counts_tr)
    hawkes_ar1 = fit_hawkes_ar1_poisson(
        counts=binned.counts,
        idx_train=idxs["train"],
        idx_val=idxs["val"],
        epochs=1500,
        patience=200,
    )
    hawkes_train_nll = hawkes_ar1_nll_from_state(counts=counts_tr, state=hawkes_ar1.model_state)
    hawkes_val_nll = hawkes_ar1_nll_from_state(counts=counts_val, state=hawkes_ar1.model_state)
    hawkes_test_nll = hawkes_ar1_nll_from_state(counts=counts_te, state=hawkes_ar1.model_state)

    metrics["baselines"] = {
        "poisson_iid": {
            "train_nll": poi.nll(counts_tr),
            "val_nll": poi.nll(counts_val),
            "test_nll": poi.nll(counts_te),
        },
        "hawkes_ar1_poisson": {
            "best_val_nll": hawkes_ar1.best_val_nll,
            "train_nll": hawkes_train_nll,
            "val_nll": hawkes_val_nll,
            "test_nll": hawkes_test_nll,
        },
    }

    # Baseline (supervised next-month multi-label shock classifier)
    _, logit_res = fit_logistic_multilabel(cfg=cfg, ds=ds, hidden=None)
    _, mlp_res = fit_logistic_multilabel(cfg=cfg, ds=ds, hidden=64)
    metrics["baselines"]["logistic_multilabel"] = {"test_macro_auc": logit_res.macro_auc}
    metrics["baselines"]["mlp_multilabel"] = {"test_macro_auc": mlp_res.macro_auc}
    metrics["baselines"]["var1_threshold_proxy"] = {"test_macro_auc": eval_var1_nextmonth_auc(cfg=cfg, ds=ds)}

    write_json(out_dir / "metrics.json", metrics)
    print(f"Wrote: {out_dir / 'metrics.json'}")

    # Lightweight plot: event type frequency.
    counts = np.bincount(np.array(ds.types, dtype=int), minlength=len(ds.type_names))
    save_bar_top_types(
        out_path=Path(out_dir) / "top_event_types.png",
        type_names=ds.type_names,
        counts=counts,
    )
    print(f"Wrote: {out_dir / 'top_event_types.png'}")


if __name__ == "__main__":
    main()


