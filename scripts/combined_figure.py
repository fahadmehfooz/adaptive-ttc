"""Build one paper-ready composite figure summarizing all three findings.

Panel A (scale):       compute saving vs model size on GSM8K, per method.
Panel B (transfer):    saving by target task (1.5B), confidence rule vs trained gate.
Panel C (calibration): ECE of the agreement signal (K=16) per model x task.

Reads only existing JSON artifacts (no GPU, no rollout re-read):
  outputs/results/scale_summary.json
  outputs/results/transfer_matrix_1.5b.json
  outputs/results/calibration.json
Writes outputs/figures/combined_results.png. See CLAUDE.md S8.
"""
import json
import os

from src import config
from src.logutil import log


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    config.ensure_dirs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = _load(os.path.join(config.RESULTS_DIR, "scale_summary.json"))
    transfer = _load(os.path.join(config.RESULTS_DIR, "transfer_matrix_1.5b.json"))
    calib = _load(os.path.join(config.RESULTS_DIR, "calibration.json"))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # --- Panel A: scale ---
    models = sorted(scale, key=lambda m: scale[m]["size_b"])
    sizes = [scale[m]["size_b"] for m in models]
    for method, color, label in [
        ("adaptive-confidence", "tab:blue", "confidence"),
        ("adaptive-agreement", "tab:green", "agreement"),
        ("esc", "tab:gray", "ESC"),
    ]:
        ys = [scale[m][method] for m in models]
        axA.plot(sizes, ys, "o-", color=color, label=label)
    axA.set_xscale("log")
    axA.set_xticks(sizes)
    axA.set_xticklabels([f"{s:g}B" for s in sizes])
    axA.set_xlabel("model size")
    axA.set_ylabel("compute saving (1 − cost/K)")
    axA.set_title("(a) Saving grows with competence (GSM8K)")
    axA.set_ylim(0, 1)
    axA.legend(fontsize=8, loc="upper left")

    # --- Panel B: transfer (1.5B), confidence vs trained gate ---
    # order targets as in-dist GSM8K, then MATH, then BBH
    name_order = ["gsm8k_qwen-1.5b.jsonl", "math500_qwen-1.5b.jsonl", "bbh_qwen-1.5b.jsonl"]
    labels = ["GSM8K\n(in-dist)", "MATH-500", "BBH"]
    present = [(n, l) for n, l in zip(name_order, labels) if n in transfer]
    xs = range(len(present))
    conf = [transfer[n]["methods"]["adaptive-confidence"]["savings_vs_full"] or 0 for n, _ in present]
    trained = [transfer[n]["methods"]["adaptive-trained"]["savings_vs_full"] or 0 for n, _ in present]
    w = 0.38
    axB.bar([x - w / 2 for x in xs], conf, w, color="tab:blue", label="confidence (training-free)")
    axB.bar([x + w / 2 for x in xs], trained, w, color="tab:orange", label="trained gate")
    axB.set_xticks(list(xs))
    axB.set_xticklabels([l for _, l in present], fontsize=8)
    axB.set_ylabel("compute saving")
    axB.set_title("(b) Learned gate collapses OOD (1.5B)")
    axB.legend(fontsize=8, loc="upper right")

    # --- Panel C: calibration ECE at K=16 ---
    k = 16
    rows = [r for r in calib if r["k"] == k]
    rows.sort(key=lambda r: (r["dataset"], r["model"]))
    cats = [f"{r['dataset']}\n{r['model'].replace('qwen-', '')}" for r in rows]
    eces = [r["ece_raw"] for r in rows]
    colors = ["tab:red" if r["dataset"] == "gsm8k" else "tab:purple" for r in rows]
    axC.bar(range(len(rows)), eces, color=colors)
    axC.axhline(0.1, ls="--", color="gray", lw=1)
    axC.text(len(rows) - 0.5, 0.105, "well-calibrated (<.10)", ha="right", va="bottom",
             fontsize=7, color="gray")
    axC.set_xticks(range(len(rows)))
    axC.set_xticklabels(cats, fontsize=7)
    axC.set_ylabel("ECE of agreement signal (K=16)")
    axC.set_title("(c) Miscalibration is task-shaped")

    fig.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "combined_results.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
