from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from nhp_torch.config.io import Config
from nhp_torch.data.events import EventDataset
from nhp_torch.models.nhp import NeuralHawkes
from nhp_torch.training.loss import nhp_loglikelihood_mc
from nhp_torch.training.progress import tqdm_wrap
from nhp_torch.utils import write_json


@dataclass(frozen=True)
class TrainResult:
    best_checkpoint: Path
    history: list[dict[str, Any]]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensorize(ds: EventDataset, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.tensor(ds.split_mask[split], dtype=torch.bool, device=device)
    times = torch.tensor(ds.times, dtype=torch.float32, device=device)[mask]
    types = torch.tensor(ds.types, dtype=torch.long, device=device)[mask]
    marks = torch.tensor(ds.marks, dtype=torch.float32, device=device)[mask]
    if times.numel() == 0:
        raise ValueError(f"No events in split={split!r}. Consider lowering threshold_z.")
    return times, types, marks


def train_model(cfg: Config, ds: EventDataset, *, ckpt_dir: Path) -> TrainResult:
    train_cfg = cfg.train
    model_cfg = cfg.model

    _set_seed(int(train_cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K = len(ds.type_names)
    model = NeuralHawkes(
        num_types=K,
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        use_marks=bool(model_cfg["use_marks"]),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    times_tr, types_tr, marks_tr = _tensorize(ds, "train", device)
    times_val, types_val, marks_val = _tensorize(ds, "val", device)

    t_start = 0.0
    # global end time based on underlying month index range (not split range)
    t_end = float(max(ds.times)) + 1.0
    mc = int(train_cfg["mc_samples_per_interval"])

    best_val = -float("inf")
    best_path = ckpt_dir / "best.pt"
    patience = int(train_cfg["patience"])
    bad = 0
    history: list[dict[str, Any]] = []

    pbar = tqdm_wrap(
        range(int(train_cfg["epochs"])),
        desc="training",
        unit="epoch",
        leave=True,
    )
    for epoch in pbar:
        model.train()
        opt.zero_grad(set_to_none=True)
        ll = nhp_loglikelihood_mc(
            model=model,
            times=times_tr,
            types=types_tr,
            marks=marks_tr,
            t_start=t_start,
            t_end=t_end,
            mc_samples_per_interval=mc,
        )
        loss = -ll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            ll_val = nhp_loglikelihood_mc(
                model=model,
                times=times_val,
                types=types_val,
                marks=marks_val,
                t_start=t_start,
                t_end=t_end,
                mc_samples_per_interval=mc,
            )

        rec = {"epoch": epoch, "train_ll": float(ll.item()), "val_ll": float(ll_val.item())}
        history.append(rec)

        if float(ll_val.item()) > best_val:
            best_val = float(ll_val.item())
            bad = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.raw,
                    "type_names": ds.type_names,
                    "best_val_ll": best_val,
                },
                best_path,
            )
        else:
            bad += 1

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                train_ll=f"{rec['train_ll']:.2f}",
                val_ll=f"{rec['val_ll']:.2f}",
                best_val=f"{best_val:.2f}",
                bad=bad,
            )
        elif epoch % 10 == 0:
            print(
                f"epoch={epoch} train_ll={rec['train_ll']:.3f} val_ll={rec['val_ll']:.3f} best_val={best_val:.3f}"
            )

        if bad >= patience:
            print(f"Early stopping at epoch={epoch} best_val_ll={best_val:.3f}")
            break

    write_json(ckpt_dir / "train_history.json", history)
    print(f"Wrote: {ckpt_dir / 'train_history.json'}")
    print(f"Best checkpoint: {best_path}")

    return TrainResult(best_checkpoint=best_path, history=history)


