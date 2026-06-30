"""S7 — Calibration analysis of the confidence signal across scale x task.

The adaptive-SC confidence signal is the majority *agreement fraction* over the
first-k samples (gate.features(...)["agreement"]). A stopping rule that trusts this
signal is only sound if the signal is *calibrated*: when agreement says 0.8, the
majority answer should be correct ~80% of the time.

This script, for each rollout file and each decision budget k:
  - collects (confidence = agreement over first-k, label = majority-of-first-k correct)
  - reports raw ECE (src.calibration.ece)
  - fits a scalar temperature on the confidence logits and reports post-scaling ECE + T

Outputs:
  - outputs/results/calibration.json      (full grid)
  - outputs/gate/calibration_temperatures.json  (fitted T per model x task x k)
  - outputs/figures/calibration_reliability.png  (reliability diagrams, k=16)

CPU only, no GPU. See CLAUDE.md S7.
"""
import argparse
import glob
import json
import math
import os

from src import config
from src.eval import load_rollouts, _majority_correct
from src import gate
from src.calibration import ece, temperature_scale
from src.logutil import log


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _parse_name(path):
    """outputs/rollouts/<dataset>_<model>.jsonl -> (dataset, model)."""
    base = os.path.basename(path)[: -len(".jsonl")] if path.endswith(".jsonl") else os.path.basename(path)
    dataset, _, model = base.partition("_")
    return dataset, model


def analyze_file(path, budgets):
    rows = load_rollouts(path)
    dataset, model = _parse_name(path)
    out = []
    diagrams = {}  # k -> (confidences, labels) for the reliability figure
    for k in budgets:
        confs, labels = [], []
        for r in rows:
            head = r["samples"][:k]
            confs.append(gate.features(head)["agreement"])
            labels.append(int(_majority_correct(head)))
        n = len(rows)
        ece_raw = ece(confs, labels)
        # temperature scaling on the confidence logits
        logits = [_logit(c) for c in confs]
        T = temperature_scale(logits, labels)
        scaled = [_sigmoid(z / T) for z in logits]
        ece_scaled = ece(scaled, labels)
        out.append({
            "dataset": dataset, "model": model, "k": k, "n": n,
            "mean_confidence": sum(confs) / n,
            "accuracy": sum(labels) / n,
            "ece_raw": ece_raw,
            "temperature": T,
            "ece_scaled": ece_scaled,
        })
        diagrams[k] = (confs, labels)
        log(f"{dataset:8s} {model:10s} k={k:2d}  ECE {ece_raw:.3f} -> {ece_scaled:.3f} (T={T:.2f})  acc {sum(labels)/n:.3f}")
    return out, diagrams


def reliability_figure(per_file_diagrams, k, out_path, n_bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(name, d[k]) for name, d in per_file_diagrams.items() if k in d]
    if not items:
        return None
    ncol = min(3, len(items))
    nrow = math.ceil(len(items) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow), squeeze=False)
    for ax, (name, (confs, labels)) in zip([a for row in axes for a in row], items):
        bins_c = [[] for _ in range(n_bins)]
        bins_y = [[] for _ in range(n_bins)]
        for c, y in zip(confs, labels):
            idx = min(n_bins - 1, int(c * n_bins))
            bins_c[idx].append(c)
            bins_y[idx].append(y)
        xs, ys = [], []
        for bc, by in zip(bins_c, bins_y):
            if bc:
                xs.append(sum(bc) / len(bc))
                ys.append(sum(by) / len(by))
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
        ax.plot(xs, ys, "o-", color="tab:blue", label="observed")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("confidence (agreement)")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    # hide any unused axes
    for ax in [a for row in axes for a in row][len(items):]:
        ax.axis("off")
    fig.suptitle(f"Reliability of the agreement signal (k={k})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="*", default=None,
                    help="rollout JSONLs; default = all real gsm8k/bbh/math500 files")
    ap.add_argument("--budgets", type=int, nargs="*", default=[4, 8, 16])
    args = ap.parse_args()

    config.ensure_dirs()
    if args.rollouts:
        files = args.rollouts
    else:
        files = sorted(
            f for f in glob.glob(os.path.join(config.ROLLOUTS_DIR, "*.jsonl"))
            if not os.path.basename(f).startswith("toy")
        )
    log(f"calibration analysis over {len(files)} files, budgets={args.budgets}")

    grid = []
    per_file_diagrams = {}
    for path in files:
        rows_out, diagrams = analyze_file(path, args.budgets)
        grid.extend(rows_out)
        dataset, model = _parse_name(path)
        per_file_diagrams[f"{dataset} / {model}"] = diagrams

    results_path = os.path.join(config.RESULTS_DIR, "calibration.json")
    with open(results_path, "w") as f:
        json.dump(grid, f, indent=2)
    log(f"wrote {results_path}")

    temps = {
        f"{r['dataset']}_{r['model']}_k{r['k']}": r["temperature"] for r in grid
    }
    temps_path = os.path.join(config.GATE_DIR, "calibration_temperatures.json")
    with open(temps_path, "w") as f:
        json.dump(temps, f, indent=2)
    log(f"wrote {temps_path}")

    fig_path = os.path.join(config.FIGURES_DIR, "calibration_reliability.png")
    saved = reliability_figure(per_file_diagrams, k=max(args.budgets), out_path=fig_path)
    if saved:
        log(f"wrote {saved}")

    # markdown summary to stdout
    print("\n| dataset | model | k | n | mean_conf | acc | ECE | ECE(T) | T |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in grid:
        print(f"| {r['dataset']} | {r['model']} | {r['k']} | {r['n']} | "
              f"{r['mean_confidence']:.3f} | {r['accuracy']:.3f} | {r['ece_raw']:.3f} | "
              f"{r['ece_scaled']:.3f} | {r['temperature']:.2f} |")


if __name__ == "__main__":
    main()
