from __future__ import annotations

import argparse
from pathlib import Path

from nhp_torch.utils import read_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a simple markdown report from evaluation artifacts.")
    ap.add_argument("--metrics", default="artifacts/eval/metrics.json", help="Path to metrics.json")
    ap.add_argument("--out", default="artifacts/eval/summary.md", help="Output markdown file")
    args = ap.parse_args()

    import json

    metrics = json.loads(read_text(args.metrics))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("## Neural Hawkes evaluation summary\n")
    lines.append(f"- checkpoint: `{metrics.get('checkpoint')}`\n")
    lines.append(f"- train_ll: {metrics.get('train_ll')}\n")
    lines.append(f"- val_ll: {metrics.get('val_ll')} (best_val_ll: {metrics.get('best_val_ll')})\n")
    lines.append(f"- test_ll: {metrics.get('test_ll')}\n")
    lines.append("\n## Horizon AUC (test)\n")
    for h, obj in (metrics.get('horizon_auc') or {}).items():
        lines.append(f"- H={h}: macro_auc_mean={obj.get('macro_auc_mean')}\n")
    lines.append("\n## Baselines\n")
    baselines = metrics.get("baselines") or {}
    if "poisson_iid" in baselines:
        b = baselines["poisson_iid"]
        lines.append(f"- poisson_iid nll: train={b.get('train_nll'):.3f} val={b.get('val_nll'):.3f} test={b.get('test_nll'):.3f}\n")
    if "hawkes_ar1_poisson" in baselines:
        b = baselines["hawkes_ar1_poisson"]
        lines.append(f"- hawkes_ar1_poisson nll: train={b.get('train_nll'):.3f} val={b.get('val_nll'):.3f} test={b.get('test_nll'):.3f}\n")
    if "logistic_multilabel" in baselines:
        lines.append(f"- logistic_multilabel test_macro_auc={baselines['logistic_multilabel'].get('test_macro_auc')}\n")
    if "mlp_multilabel" in baselines:
        lines.append(f"- mlp_multilabel test_macro_auc={baselines['mlp_multilabel'].get('test_macro_auc')}\n")
    if "var1_threshold_proxy" in baselines:
        lines.append(f"- var1_threshold_proxy test_macro_auc={baselines['var1_threshold_proxy'].get('test_macro_auc')}\n")
    lines.append("\n## Plots\n")
    lines.append("- `artifacts/eval/top_event_types.png`\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()


