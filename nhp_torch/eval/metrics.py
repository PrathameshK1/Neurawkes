from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from nhp_torch.config.io import Config
from nhp_torch.data.events import EventDataset
from nhp_torch.eval.binary_eval import eval_horizon_auc
from nhp_torch.eval.event_prediction import eval_event_prediction, eval_monthly_realistic
from nhp_torch.models.nhp import NeuralHawkes
from nhp_torch.training.loss import nhp_loglikelihood_mc


def _tensorize(ds: EventDataset, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.tensor(ds.split_mask[split], dtype=torch.bool, device=device)
    times = torch.tensor(ds.times, dtype=torch.float32, device=device)[mask]
    types = torch.tensor(ds.types, dtype=torch.long, device=device)[mask]
    marks = torch.tensor(ds.marks, dtype=torch.float32, device=device)[mask]
    return times, types, marks


def evaluate_checkpoint(cfg: Config, ds: EventDataset, *, checkpoint_path: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = cfg.model
    K = len(ds.type_names)
    model = NeuralHawkes(
        num_types=K,
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        use_marks=bool(model_cfg["use_marks"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mc = int(cfg.train["mc_samples_per_interval"])
    t_start = 0.0
    t_end = float(max(ds.times)) + 1.0

    out: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "best_val_ll": float(ckpt.get("best_val_ll", float("nan"))),
        "num_types": K,
    }

    for split in ("train", "val", "test"):
        times, types, marks = _tensorize(ds, split, device)
        if times.numel() == 0:
            out[f"{split}_ll"] = None
            out[f"{split}_events"] = 0
            continue
        with torch.no_grad():
            ll = nhp_loglikelihood_mc(
                model=model,
                times=times,
                types=types,
                marks=marks,
                t_start=t_start,
                t_end=t_end,
                mc_samples_per_interval=mc,
            )
        out[f"{split}_ll"] = float(ll.item())
        out[f"{split}_events"] = int(times.numel())

    # Simple calibration: compare observed vs predicted total event count on test by integrating intensity.
    # Approx: sample M grid points across [0, t_end], integrate E[sum λ] dt.
    with torch.no_grad():
        grid = torch.linspace(0.0, t_end, steps=200, device=device)
        # Use full history for intensity path.
        times_all = torch.tensor(ds.times, dtype=torch.float32, device=device)
        types_all = torch.tensor(ds.types, dtype=torch.long, device=device)
        marks_all = torch.tensor(ds.marks, dtype=torch.float32, device=device)
        lam = model.forward_intensity_path(times_all, types_all, marks_all, query_times=grid)  # (G,K)
        total_rate = lam.sum(dim=-1)  # (G,)
        integral_total = float(torch.trapezoid(total_rate, grid).item())
    out["predicted_total_events_over_full_range"] = integral_total
    out["observed_total_events_over_full_range"] = int(len(ds.times))

    # Horizon prediction AUCs (macro mean over types with both classes present).
    horizons = list(cfg.eval.get("horizons_months", [1, 3]))
    out["horizon_auc"] = {}
    out["event_prediction"] = {}  # legacy (flattened month×type) metrics
    out["event_prediction_monthly"] = {}  # realistic month-level metrics
    for h in horizons:
        res = eval_horizon_auc(model=model, ds=ds, split="test", horizon_months=int(h))
        out["horizon_auc"][str(h)] = res.to_jsonable()
        ev = eval_event_prediction(
            model=model,
            ds=ds,
            split_for_threshold="val",
            split_for_scoring="test",
            horizon_months=int(h),
        )
        out["event_prediction"][str(h)] = ev.to_jsonable()
        real = eval_monthly_realistic(
            model=model,
            ds=ds,
            split_for_threshold="val",
            split_for_scoring="test",
            horizon_months=int(h),
        )
        out["event_prediction_monthly"][str(h)] = real.to_jsonable()

    return out


