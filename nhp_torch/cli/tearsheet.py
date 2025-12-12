from __future__ import annotations

import argparse
from pathlib import Path

from nhp_torch.config.io import load_config
from nhp_torch.data.events import build_event_dataset
from nhp_torch.eval.tearsheet import build_tearsheet
from nhp_torch.utils import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a QuantStats-like model evaluation tearsheet.")
    ap.add_argument("--config", default="nhp_torch/config/default.toml", help="Path to TOML config.")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    ap.add_argument(
        "--out-dir",
        default="artifacts/eval_tearsheet",
        help="Directory to write report + plots.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = build_event_dataset(cfg)

    out_dir = ensure_dir(args.out_dir)
    build_tearsheet(cfg=cfg, ds=ds, checkpoint_path=Path(args.checkpoint), out_dir=Path(out_dir))


if __name__ == "__main__":
    main()


