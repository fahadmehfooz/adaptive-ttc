"""S5 (headline #1): does the compute saving from adaptive stopping grow as models shrink?

Reads per-model rollouts, computes each method's iso-accuracy compute saving vs full budget,
and plots saving-vs-model-size. Expects one rollouts file per model.

Run:
  python -m scripts.scale_plot \
      --rollouts outputs/rollouts/gsm8k_qwen-0.5b.jsonl outputs/rollouts/gsm8k_qwen-1.5b.jsonl \
                 outputs/rollouts/gsm8k_qwen-7b.jsonl outputs/rollouts/gsm8k_llama-8b.jsonl
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, eval as ev

# model key (as it appears in the rollouts filename) -> billions of params
MODEL_SIZE = {"qwen-0.5b": 0.5, "qwen-1.5b": 1.5, "qwen-7b": 7.0, "llama-8b": 8.0}

# adaptive methods we report saving for (best point at iso-accuracy)
METHODS = ["esc", "adaptive-confidence", "adaptive-agreement", "adaptive-trained"]


def size_from_path(p):
    name = os.path.splitext(os.path.basename(p))[0]  # e.g. gsm8k_qwen-1.5b
    for key, size in MODEL_SIZE.items():
        if name.endswith(key):
            return key, size
    return name, None


def best_saving(rows, method, k0, kmax):
    """Iso-accuracy compute saving of `method` vs full budget (1 - cost/kmax)."""
    full_acc = ev.fixed_budget(rows, kmax)["accuracy"]
    pts = ev.baselines_and_adaptive(rows, k0=k0, kmax=kmax)
    mpts = [p for p in pts if p["policy"] == method and p["accuracy"] >= full_acc - 0.005]
    if not mpts:
        return None
    return 1 - min(p["mean_cost"] for p in mpts) / kmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=config.SAMPLING["n"])
    args = ap.parse_args()
    config.ensure_dirs()

    series = {m: [] for m in METHODS}  # method -> [(size, saving)]
    table = {}
    for path in args.rollouts:
        key, size = size_from_path(path)
        if size is None:
            print(f"skip (unknown model size): {path}")
            continue
        rows = ev.load_rollouts(path)
        table[key] = {"size_b": size, "full_acc": round(ev.fixed_budget(rows, args.kmax)["accuracy"], 3)}
        for m in METHODS:
            s = best_saving(rows, m, args.k0, args.kmax)
            table[key][m] = None if s is None else round(s, 3)
            if s is not None:
                series[m].append((size, s))

    out_json = os.path.join(config.RESULTS_DIR, "scale_summary.json")
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m, pts in series.items():
        if not pts:
            continue
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=m)
    ax.set_xscale("log")
    ax.set_xlabel("model size (B params, log)")
    ax.set_ylabel("compute saved at iso-accuracy")
    ax.set_title("Adaptive saving vs model size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(config.FIGURES_DIR, "scale_saving.png")
    fig.savefig(out_png, dpi=150)

    print(f"scale summary -> {out_json}")
    print(f"figure        -> {out_png}")
    for key, d in sorted(table.items(), key=lambda kv: kv[1]["size_b"]):
        print(f"  {key:12} ({d['size_b']}B) full_acc={d['full_acc']}  "
              + "  ".join(f"{m}={d[m]}" for m in METHODS))


if __name__ == "__main__":
    main()
