"""S8 (headline #2 figure): grouped-bar of iso-accuracy compute saving per method × target,
one panel per model scale. Reads the per-scale matrices written by scripts.transfer_matrix.

Run:
  python -m scripts.transfer_plot \
      --matrices outputs/results/transfer_matrix_0.5b.json outputs/results/transfer_matrix_1.5b.json \
      --scales 0.5B 1.5B
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.logutil import log

METHODS = ["esc", "adaptive-confidence", "adaptive-agreement", "adaptive-trained"]
LABELS = {"esc": "ESC", "adaptive-confidence": "confidence",
          "adaptive-agreement": "agreement", "adaptive-trained": "trained gate"}


def target_label(fn):
    base = fn.replace(".jsonl", "")
    for ds in ("gsm8k", "math500", "bbh"):
        if base.startswith(ds):
            return {"gsm8k": "GSM8K (in-dist)", "math500": "MATH-500", "bbh": "BBH"}[ds]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", required=True)
    ap.add_argument("--scales", nargs="+", required=True, help="labels aligned with --matrices")
    args = ap.parse_args()
    config.ensure_dirs()
    assert len(args.matrices) == len(args.scales), "one scale label per matrix"
    log(f"transfer_plot START: {len(args.matrices)} scale panel(s)")

    fig, axes = plt.subplots(1, len(args.matrices), figsize=(6 * len(args.matrices), 4.2), squeeze=False)
    for pi, (mpath, scale) in enumerate(zip(args.matrices, args.scales)):
        ax = axes[0][pi]
        log(f"[{pi + 1}/{len(args.matrices)}] {scale}: reading {os.path.basename(mpath)}")
        d = json.load(open(mpath))
        # Order targets: in-dist GSM8K first, then OOD.
        targets = sorted(d.keys(), key=lambda t: (0 if t.startswith("gsm8k") else 1, t))
        x = range(len(targets))
        w = 0.2
        for mi, m in enumerate(METHODS):
            vals = [(d[t]["methods"].get(m, {}).get("savings_vs_full") or 0.0) for t in targets]
            ax.bar([xi + (mi - 1.5) * w for xi in x], vals, w, label=LABELS[m])
        ax.set_xticks(list(x))
        ax.set_xticklabels([target_label(t) for t in targets], rotation=10)
        ax.set_ylabel("compute saved at iso-accuracy")
        ax.set_title(f"Qwen2.5-{scale}: transfer of stopping rules\n(gate calibrated on GSM8K)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "transfer_savings.png")
    fig.savefig(out, dpi=150)
    log(f"transfer_plot DONE figure -> {out}")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
