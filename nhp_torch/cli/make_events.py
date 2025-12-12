from __future__ import annotations

import argparse
from pathlib import Path

from nhp_torch.config.io import load_config
from nhp_torch.data.events import build_event_dataset
from nhp_torch.utils import ensure_dir, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Build event dataset artifacts from CSV.")
    ap.add_argument(
        "--config",
        default="nhp_torch/config/default.toml",
        help="Path to TOML config.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = ensure_dir(cfg.paths["artifacts_dir"])
    out_dir = ensure_dir(artifacts_dir / "events")

    ds = build_event_dataset(cfg)

    # Store as JSON for transparency (small dataset). For larger datasets we'd use parquet/npz.
    write_json(out_dir / "events.json", ds.to_jsonable())
    (out_dir / "meta.txt").write_text(ds.describe(), encoding="utf-8")

    print(f"Wrote: {out_dir / 'events.json'}")
    print(f"Wrote: {out_dir / 'meta.txt'}")


if __name__ == "__main__":
    main()


