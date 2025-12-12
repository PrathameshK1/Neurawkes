from __future__ import annotations

import argparse
from pathlib import Path

from nhp_torch.config.io import load_config
from nhp_torch.data.events import build_event_dataset
from nhp_torch.training.trainer import train_model
from nhp_torch.utils import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Neural Hawkes model (PyTorch).")
    ap.add_argument(
        "--config",
        default="nhp_torch/config/default.toml",
        help="Path to TOML config.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = ensure_dir(cfg.paths["artifacts_dir"])
    ckpt_dir = ensure_dir(artifacts_dir / "checkpoints")

    ds = build_event_dataset(cfg)
    train_model(cfg, ds, ckpt_dir=Path(ckpt_dir))


if __name__ == "__main__":
    main()


