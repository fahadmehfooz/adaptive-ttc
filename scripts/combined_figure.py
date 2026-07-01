"""Build one paper-ready composite figure summarizing the (post-critique) findings.

Panel A (scale):       confidence saving vs model size on the 7B-matched GSM8K id subset, with
                       bootstrap 95% CI error bars, plus oracle-ceiling and random-stop reference.
Panel B (transfer):    saving by target task (1.5B) under MATCHED mechanics — agreement@k0 vs
                       trained@k0 (2-stage) — with CI error bars; ties shown honestly.
Panel C (calibration): ECE of the agreement signal (K=16) per model x task, with 95% CI error bars.

Reads outputs/results/analysis.json (+ scale_matched) and outputs/results/calibration.json.
Writes outputs/figures/combined_results.png. CPU only.
"""
import json
import os

from src import config
from src.logutil import log


def _load(path):
    with open(path) as f:
        return json.load(f)


def _err(d):
    """(value, [lo_err, hi_err]) for asymmetric error bars from a {saving, ci95} dict."""
    s = d.get("saving")
    lo, hi = d.get("ci95", [None, None])
    if s is None or lo is None:
        return None, None
    return s, [[max(0.0, s - lo)], [max(0.0, hi - s)]]


def main():
    config.ensure_dirs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = _load(os.path.join(config.RESULTS_DIR, "analysis.json"))
    calib = _load(os.path.join(config.RESULTS_DIR, "calibration.json"))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14, 4.2))

    # --- Panel A: scale (matched 128-id subset) ---
    sm = A["scale_matched"]["cells"]
    order = [m for m in ("qwen-0.5b", "qwen-1.5b", "qwen-7b") if m in sm]
    sizes = [0.5, 1.5, 7.0][: len(order)]
    conf_v, conf_e = [], [[], []]
    orc_v, rnd_v = [], []
    for m in order:
        c = sm[m]
        v, e = _err(c["incremental"]["confidence"])
        conf_v.append(v if v is not None else 0)
        conf_e[0].append(e[0][0] if e else 0)
        conf_e[1].append(e[1][0] if e else 0)
        orc_v.append(c["oracle"]["saving"])
        rnd_v.append(c["random_stop"]["saving_at_iso_acc"] or 0)
    axA.errorbar(sizes, conf_v, yerr=conf_e, fmt="o-", color="tab:blue", capsize=4,
                 label="confidence (incremental)")
    axA.plot(sizes, orc_v, "s--", color="tab:green", label="oracle ceiling")
    axA.plot(sizes, rnd_v, "^:", color="tab:gray", label="random-stop control")
    axA.set_xscale("log")
    axA.set_xticks(sizes)
    axA.set_xticklabels([f"{s:g}B" for s in sizes])
    axA.set_xlabel("model size (GSM8K, matched 128 ids)")
    axA.set_ylabel("compute saving (1 − cost/K)")
    axA.set_title("(a) Saving jumps at 7B (95% CI)")
    axA.set_ylim(0, 1)
    axA.legend(fontsize=7, loc="upper left")

    # --- Panel B: paired Δ (trained − agreement), same problems, 95% CI + equivalence band ---
    cells = A["cells"]
    order_t = [("gsm8k_qwen-0.5b", "GSM8K\n0.5B"), ("gsm8k_qwen-1.5b", "GSM8K\n1.5B"),
               ("math500_qwen-1.5b", "MATH\n1.5B"), ("bbh_qwen-1.5b", "BBH\n1.5B")]
    present = [(k, l) for k, l in order_t if k in cells and
               cells[k].get("paired_trained_minus_agreement", {}).get("delta") is not None]
    xs = list(range(len(present)))
    deltas, errs = [], [[], []]
    for k, _ in present:
        pd = cells[k]["paired_trained_minus_agreement"]
        d, (lo, hi) = pd["delta"], pd["ci95"]
        deltas.append(d)
        errs[0].append(d - lo)
        errs[1].append(hi - d)
    axB.axhspan(-0.05, 0.05, color="tab:green", alpha=0.12, label="±0.05 equivalence margin")
    axB.axhline(0, color="gray", lw=1)
    axB.errorbar(xs, deltas, yerr=errs, fmt="o", color="tab:purple", capsize=4)
    axB.set_xticks(xs)
    axB.set_xticklabels([l for _, l in present], fontsize=8)
    axB.set_ylabel("Δ saving: trained − agreement (paired)")
    axB.set_title("(b) All comparisons inconclusive (95% CI ≫ margin)")
    axB.legend(fontsize=7, loc="upper right")

    # --- Panel C: calibration ECE at K=16 with CIs ---
    k = 16
    rows = [r for r in calib if r["k"] == k]
    rows.sort(key=lambda r: (r["dataset"], r["model"]))
    cats = [f"{r['dataset']}\n{r['model'].replace('qwen-', '')}" for r in rows]
    eces = [r["ece_raw"] for r in rows]
    errs = [[r["ece_raw"] - r["ece_raw_ci95"][0] for r in rows],
            [r["ece_raw_ci95"][1] - r["ece_raw"] for r in rows]]
    colors = ["tab:red" if r["dataset"] == "gsm8k" else "tab:purple" for r in rows]
    axC.bar(range(len(rows)), eces, yerr=errs, capsize=3, color=colors, label="raw ECE")
    iso = [r.get("ece_isotonic") for r in rows]
    axC.plot(range(len(rows)), iso, "kD", markersize=6, label="ECE after isotonic")
    axC.axhline(0.1, ls="--", color="gray", lw=1)
    axC.set_xticks(range(len(rows)))
    axC.set_xticklabels(cats, fontsize=7)
    axC.set_ylabel("ECE of agreement signal (K=16)")
    axC.set_title("(c) Raw ECE = share≠prob; monotone map removes it")
    axC.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "combined_results.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
