from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_bar_top_types(
    *,
    out_path: Path,
    type_names: list[str],
    counts: np.ndarray,
    top_n: int = 15,
) -> None:
    idx = np.argsort(-counts)[:top_n]
    names = [type_names[i] for i in idx]
    vals = counts[idx]

    plt.figure(figsize=(10, 5))
    plt.barh(range(len(vals)), vals[::-1])
    plt.yticks(range(len(vals)), [names[i] for i in range(len(names) - 1, -1, -1)], fontsize=7)
    plt.xlabel("count")
    plt.title("Top event types by count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


