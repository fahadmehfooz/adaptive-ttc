"""Authoritative saving analysis: matched mechanics + bootstrap CIs + yardsticks (fast).

Addresses CRITIQUE_LOG iteration 1:
  #1 mechanism confound — signals compared HEAD-TO-HEAD under matched 2-stage mechanics @ k0
     (same saving ceiling 1-k0/K); incremental confidence & ESC reported as a separate,
     clearly-labelled group (different ceiling). We never rank across mechanism groups.
  #2 no CIs — percentile bootstrap 95% CIs over problems for every saving cell, and the whole
     operating-point selection is bootstrapped (captures selection variance).
  #4 no yardstick — oracle upper bound + random-stop control per cell; saving as %-of-oracle.

Speed: per-problem (cost, correct) arrays are precomputed ONCE per method/threshold; the bootstrap
then only resamples column indices and averages (numpy). B=2000 runs in seconds.

Groups & signals:
  head-to-head @ k0 (2-stage, ceiling 1-k0/K):  agreement-share   vs  trained-gate-prob
  incremental (ceiling 1-min_k/K):              confidence(=agreement, incremental)  vs  ESC
  yardsticks:                                   oracle (matched-accuracy)  ,  random-stop (control)

Writes outputs/results/analysis.json. CPU only.
"""
import argparse
import glob
import json
import os

import numpy as np

from src import config, eval as ev, gate as gatemod
from src.logutil import log

MODEL_ORDER = {"qwen-0.5b": 0, "qwen-1.5b": 1, "qwen-7b": 2, "llama-8b": 3}
TOL = 0.01


def parse_name(fn):
    base = os.path.splitext(os.path.basename(fn))[0]
    for m in MODEL_ORDER:
        if base.endswith(m):
            return base[: -(len(m) + 1)], m
    return base, "?"


# ---- per-problem precompute -------------------------------------------------

def _full_correct(rows, kmax):
    return np.array([ev._majority_correct(r["samples"][:kmax]) for r in rows], dtype=float)


def _twostage_arrays(rows, signal, head_correct, full_correct, ts, k0, kmax):
    """(cost, correct) [T,n] for a 2-stage rule: stop@k0 if signal>=t else go to kmax."""
    sig = signal[None, :]           # (1,n)
    T = ts[:, None]                 # (Tt,1)
    stop = sig >= T                 # (Tt,n)
    cost = np.where(stop, float(k0), float(kmax))
    correct = np.where(stop, head_correct[None, :], full_correct[None, :])
    return cost, correct


def _incremental_arrays(rows, should_stop_factory, ts, min_k, kmax):
    """(cost, correct) [T,n] for an incremental rule swept over thresholds ts."""
    n = len(rows)
    cost = np.zeros((len(ts), n))
    correct = np.zeros((len(ts), n))
    for ti, t in enumerate(ts):
        stop = should_stop_factory(t)
        for i, r in enumerate(rows):
            c, k = ev._incremental(r["samples"], kmax, stop, min_k=min_k)
            correct[ti, i] = c
            cost[ti, i] = k
    return cost, correct


def _boot_saving(cost, correct, full_correct, kmax, B, seed, tol=TOL):
    """Percentile-bootstrap the iso-accuracy saving of a (cost,correct)[T,n] method.
    Returns (point, lo, hi, boot_valid)."""
    rng = np.random.default_rng(seed)
    n = full_correct.shape[0]

    def saving(idx):
        full = full_correct[idx].mean()
        accs = correct[:, idx].mean(axis=1)
        costs = cost[:, idx].mean(axis=1)
        ok = accs >= full - tol
        if not ok.any():
            return None
        return 1.0 - costs[ok].min() / kmax

    point = saving(np.arange(n))
    vals = []
    for _ in range(B):
        v = saving(rng.integers(0, n, size=n))
        if v is not None:
            vals.append(v)
    if not vals:
        return (point, None, None, 0)
    vals = np.sort(np.array(vals))
    lo = float(vals[int(0.025 * len(vals))])
    hi = float(vals[min(len(vals) - 1, int(0.975 * len(vals)))])
    return (None if point is None else round(float(point), 4),
            round(lo, 4), round(hi, 4), len(vals))


def analyze_cell(rows, model, kmax, k0, min_k, B, seed, gate_dir):
    gp = os.path.join(gate_dir, f"gsm8k_{model}.joblib")
    gate_model = gatemod.TrainedGate.load(gp) if os.path.exists(gp) else None

    full_correct = _full_correct(rows, kmax)
    head_correct = np.array([ev._majority_correct(r["samples"][:k0]) for r in rows], dtype=float)
    full_acc = float(full_correct.mean())
    cell = {"n": len(rows), "kmax": kmax, "k0": k0, "min_k": min_k,
            "full_acc": round(full_acc, 3),
            "gate": os.path.basename(gp) if gate_model else None,
            "ceiling_2stage": round(1 - k0 / kmax, 3),
            "ceiling_incremental": round(1 - min_k / kmax, 3),
            "head_to_head_2stage": {}, "incremental": {}}

    # --- head-to-head @ k0 (matched mechanism) ---
    agr_sig = np.array([ev.gate.features(r["samples"][:k0])["agreement"] for r in rows])
    ts_agr = np.array([i / 10 for i in range(2, 11)])
    cost, corr = _twostage_arrays(rows, agr_sig, head_correct, full_correct, ts_agr, k0, kmax)
    p, lo, hi, nv = _boot_saving(cost, corr, full_correct, kmax, B, seed)
    cell["head_to_head_2stage"]["agreement"] = {"saving": p, "ci95": [lo, hi], "boot_valid": nv}

    if gate_model is not None:
        gp_sig = np.array([gate_model.predict_proba(ev.gate.features(r["samples"][:k0])) for r in rows])
        ts_tr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        cost, corr = _twostage_arrays(rows, gp_sig, head_correct, full_correct, ts_tr, k0, kmax)
        p, lo, hi, nv = _boot_saving(cost, corr, full_correct, kmax, B, seed)
        cell["head_to_head_2stage"]["trained"] = {"saving": p, "ci95": [lo, hi], "boot_valid": nv}

    # --- incremental group ---
    ts_conf = np.array([i / 10 for i in range(5, 11)])
    cost, corr = _incremental_arrays(
        rows, lambda t: (lambda d, t=t: ev.gate.features(d)["agreement"] >= t),
        ts_conf, min_k, kmax)
    p, lo, hi, nv = _boot_saving(cost, corr, full_correct, kmax, B, seed)
    cell["incremental"]["confidence"] = {"saving": p, "ci95": [lo, hi], "boot_valid": nv}

    # ESC: single config, incremental
    esc_cost = np.zeros((1, len(rows)))
    esc_corr = np.zeros((1, len(rows)))
    for i, r in enumerate(rows):
        c, k = ev._incremental(r["samples"], kmax, lambda d: ev._esc_stop(d, 4), min_k=4)
        esc_corr[0, i] = c
        esc_cost[0, i] = k
    p, lo, hi, nv = _boot_saving(esc_cost, esc_corr, full_correct, kmax, B, seed)
    cell["incremental"]["esc"] = {"saving": p, "ci95": [lo, hi], "boot_valid": nv}

    # --- yardsticks ---
    orc_k = np.zeros((1, len(rows)))
    for i, r in enumerate(rows):
        s = r["samples"]
        final, _ = ev.gate.majority([x["answer"] for x in s[:kmax]])
        k = kmax
        for j in range(1, kmax + 1):
            ans, _ = ev.gate.majority([x["answer"] for x in s[:j]])
            if ans == final and ans is not None:
                k = j
                break
        orc_k[0, i] = k
    orc_corr = full_correct[None, :]
    p, lo, hi, _ = _boot_saving(orc_k, orc_corr, full_correct, kmax, B, seed)
    cell["oracle"] = {"saving": p, "ci95": [lo, hi]}

    rcurve = ev.random_stop_curve(rows, kmax=kmax, min_k=min_k, seed=seed)
    ok = [q for q in rcurve if q["accuracy"] >= full_acc - TOL]
    rsav = (round(1 - min(q["mean_cost"] for q in ok) / kmax, 4) if ok else None)
    cell["random_stop"] = {"saving_at_iso_acc": rsav}

    # %-of-oracle for the best real method
    for group in ("head_to_head_2stage", "incremental"):
        for d in cell[group].values():
            d["frac_of_oracle"] = (round(d["saving"] / p, 3)
                                   if d["saving"] is not None and p else None)
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="*", default=None)
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("-B", "--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-scale", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    files = args.rollouts or sorted(
        f for f in glob.glob(os.path.join(config.ROLLOUTS_DIR, "*.jsonl"))
        if not os.path.basename(f).startswith("toy"))

    log(f"analysis START: {len(files)} cells, k0={args.k0}, min_k={args.min_k}, B={args.bootstrap}")
    out = {"config": {"k0": args.k0, "min_k": args.min_k, "B": args.bootstrap,
                      "seed": args.seed, "tol": TOL}, "cells": {}}
    rows_by = {}
    for fi, f in enumerate(files, 1):
        ds, model = parse_name(f)
        rows = ev.load_rollouts(f)
        kmax = len(rows[0]["samples"])
        rows_by[(ds, model)] = rows
        log(f"[{fi}/{len(files)}] {ds}/{model} n={len(rows)} ...")
        out["cells"][f"{ds}_{model}"] = analyze_cell(
            rows, model, kmax, args.k0, args.min_k, args.bootstrap, args.seed, config.GATE_DIR)

    if args.match_scale:
        gk = [(m, rows_by[("gsm8k", m)]) for m in ("qwen-0.5b", "qwen-1.5b", "qwen-7b")
              if ("gsm8k", m) in rows_by]
        if any(m == "qwen-7b" for m, _ in gk):
            keep = {r["id"] for m, rs in gk if m == "qwen-7b" for r in rs}
            log(f"match-scale: restricting GSM8K to {len(keep)} shared 7B ids")
            out["scale_matched"] = {"n_ids": len(keep), "cells": {}}
            for m, rs in gk:
                sub = [r for r in rs if r["id"] in keep]
                kmax = len(sub[0]["samples"])
                out["scale_matched"]["cells"][m] = analyze_cell(
                    sub, m, kmax, args.k0, args.min_k, args.bootstrap, args.seed, config.GATE_DIR)

    dest = os.path.join(config.RESULTS_DIR, "analysis.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    log(f"analysis DONE -> {dest}")

    def fmt(d):
        s, (lo, hi) = d.get("saving"), d.get("ci95", [None, None])
        return "n.r." if s is None else f"{s:.3f} [{lo:.3f},{hi:.3f}]"
    print("\n=== head-to-head @ k0 (2-stage, matched mechanism) + incremental + yardsticks ===")
    print("| cell | n | full_acc | agree@k0 | trained@k0 | conf(inc) | esc | oracle | rand |")
    print("|---|--:|--:|---|---|---|---|---|--:|")
    for name, c in out["cells"].items():
        h, inc = c["head_to_head_2stage"], c["incremental"]
        r = c["random_stop"]["saving_at_iso_acc"]
        print(f"| {name} | {c['n']} | {c['full_acc']} | {fmt(h.get('agreement',{}))} | "
              f"{fmt(h.get('trained',{}))} | {fmt(inc.get('confidence',{}))} | "
              f"{fmt(inc.get('esc',{}))} | {fmt(c['oracle'])} | {'n.r.' if r is None else f'{r:.3f}'} |")


if __name__ == "__main__":
    main()
