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

from src import config, eval as ev, gate as gatemod, stats
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


def _saving_fn(cost, correct, full_correct, kmax, tol=TOL):
    """Return idx-array -> iso-accuracy saving for a (cost,correct)[T,n] method (None if the bar is
    unreachable). Used for both marginal CIs and the paired-difference bootstrap (shared indices)."""
    def saving(idx):
        full = full_correct[idx].mean()
        accs = correct[:, idx].mean(axis=1)
        costs = cost[:, idx].mean(axis=1)
        ok = accs >= full - tol
        if not ok.any():
            return None
        return 1.0 - costs[ok].min() / kmax
    return saving


def _boot_saving(cost, correct, full_correct, kmax, B, seed, tol=TOL):
    """Percentile-bootstrap the iso-accuracy saving of a (cost,correct)[T,n] method.
    Returns (point, lo, hi, boot_valid)."""
    rng = np.random.default_rng(seed)
    n = full_correct.shape[0]
    saving = _saving_fn(cost, correct, full_correct, kmax, tol)
    point = saving(np.arange(n))
    vals = []
    for _ in range(B):
        v = saving(rng.integers(0, n, size=n))
        if v is not None:
            vals.append(v)
    if not vals or point is None:
        # no point estimate (operating point unreachable) -> do not report a CI for a non-existent value
        return (None, None, None, len(vals))
    vals = np.sort(np.array(vals))
    lo = float(vals[int(0.025 * len(vals))])
    hi = float(vals[min(len(vals) - 1, int(0.975 * len(vals)))])
    return (round(float(point), 4), round(lo, 4), round(hi, 4), len(vals))


def _conf_chainperm_ci(rows, kmax, min_k, ts_conf, B, seed):
    """Arrival-ORDER sensitivity check: CI for the incremental-confidence saving under a nested
    bootstrap that resamples problems AND permutes each problem's chain arrival order. The 16 chains
    per problem are exchangeable (iid at fixed temperature), so re-ordering them tests whether the
    result depends on *which fixed chains arrive first*. SCOPE (important, do not overstate): this
    covers only the arrival-ORDER component of decoding variance. It does NOT cover chain-IDENTITY
    variance (a fresh decoding seed yields 16 *different* chains, changing the agreement statistic the
    rule keys on) — that needs fresh multi-seed rollouts — and it says nothing about the 4-bit-vs-fp16
    precision confound, which is systematic, not stochastic. Use it to rule out an ordering artifact,
    not to claim seed-robustness or to identify a scale effect."""
    from collections import Counter
    n = len(rows)
    answers = [[x["answer"] for x in r["samples"][:kmax]] for r in rows]
    corrmap = []
    for r in rows:
        m = {}
        for x in r["samples"][:kmax]:
            a = x["answer"]
            if a is not None and a not in m:
                m[a] = bool(x["correct"])
        corrmap.append(m)
    full_correct = np.array([ev._majority_correct(r["samples"][:kmax]) for r in rows], dtype=float)
    rng = np.random.default_rng(seed)

    def problem_pts(i, order):
        seq = [answers[i][j] for j in order]
        cnt = Counter()
        agr = [0.0] * (len(seq) + 1)
        plur = [None] * (len(seq) + 1)
        nn = 0
        for k in range(1, len(seq) + 1):
            a = seq[k - 1]
            if a is not None:
                cnt[a] += 1
                nn += 1
            if nn:
                top, c = cnt.most_common(1)[0]
                agr[k], plur[k] = c / nn, top
        res = {}
        for t in ts_conf:
            sk = len(seq)
            for k in range(min_k, len(seq) + 1):
                if agr[k] >= t:
                    sk = k
                    break
            pa = plur[sk]
            res[t] = (sk, 1.0 if (pa is not None and corrmap[i].get(pa, False)) else 0.0)
        return res

    def draw_saving(idx):
        agg = {t: [0.0, 0.0] for t in ts_conf}  # t -> [cost_sum, correct_sum]
        for i in idx:
            order = rng.permutation(kmax)
            pts = problem_pts(i, order)
            for t in ts_conf:
                c, y = pts[t]
                agg[t][0] += c
                agg[t][1] += y
        full = full_correct[idx].mean()
        m = len(idx)
        best = None
        for t in ts_conf:
            acc = agg[t][1] / m
            cost = agg[t][0] / m
            if acc >= full - TOL:
                s = 1 - cost / kmax
                best = s if best is None else max(best, s)
        return best

    point = draw_saving(np.arange(n))
    vals = [v for v in (draw_saving(rng.integers(0, n, size=n)) for _ in range(B)) if v is not None]
    if not vals or point is None:
        return [None, None]
    vals = np.sort(np.array(vals))
    return [round(float(vals[int(0.025 * len(vals))]), 4),
            round(float(vals[min(len(vals) - 1, int(0.975 * len(vals)))]), 4)]


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
    agr_cost, agr_corr = _twostage_arrays(rows, agr_sig, head_correct, full_correct, ts_agr, k0, kmax)
    p, lo, hi, nv = _boot_saving(agr_cost, agr_corr, full_correct, kmax, B, seed)
    ceiling = 1 - k0 / kmax
    cell["head_to_head_2stage"]["agreement"] = {
        "saving": p, "ci95": [lo, hi], "boot_valid": nv,
        "ceiling_censored": p is not None and abs(hi - ceiling) < 1e-9}

    if gate_model is not None:
        gp_sig = np.array([gate_model.predict_proba(ev.gate.features(r["samples"][:k0])) for r in rows])
        ts_tr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        tr_cost, tr_corr = _twostage_arrays(rows, gp_sig, head_correct, full_correct, ts_tr, k0, kmax)
        p, lo, hi, nv = _boot_saving(tr_cost, tr_corr, full_correct, kmax, B, seed)
        cell["head_to_head_2stage"]["trained"] = {
            "saving": p, "ci95": [lo, hi], "boot_valid": nv,
            "ceiling_censored": p is not None and abs(hi - ceiling) < 1e-9}
        # PAIRED difference test (same problems) + TOST equivalence (fix iter2 #1)
        n = len(rows)
        pd = stats.paired_delta(
            _saving_fn(agr_cost, agr_corr, full_correct, kmax),
            _saving_fn(tr_cost, tr_corr, full_correct, kmax),
            n, B=B, seed=seed, margin=0.05)
        pd["censored"] = (cell["head_to_head_2stage"]["agreement"]["ceiling_censored"]
                          or cell["head_to_head_2stage"]["trained"]["ceiling_censored"])
        cell["paired_trained_minus_agreement"] = pd

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
    p_cons, lo, hi, _ = _boot_saving(orc_k, orc_corr, full_correct, kmax, B, seed)
    cell["oracle_consistency"] = {"saving": p_cons, "ci95": [lo, hi],
                                  "note": "stop when plurality==full-budget answer (preserves full acc)"}
    # achievability oracle: stop at earliest correct plurality (fix iter2 #3)
    ach = ev.oracle_achievable(rows, kmax)
    cell["oracle_achievable"] = {"saving": ach["saving"], "accuracy": round(ach["accuracy"], 3),
                                 "note": "stop at earliest correct plurality; acc may exceed full-budget"}
    cell["oracle"] = cell["oracle_consistency"]  # back-compat key

    rcurve = ev.random_stop_curve(rows, kmax=kmax, min_k=min_k, seed=seed)
    ok = [q for q in rcurve if q["accuracy"] >= full_acc - TOL]
    rsav = (round(1 - min(q["mean_cost"] for q in ok) / kmax, 4) if ok else None)
    cell["random_stop"] = {"saving_at_iso_acc": rsav}

    # %-of-oracle vs the CONSISTENCY oracle — the correct iso-accuracy ceiling for our metric
    # (our saving is defined at full-budget accuracy; the achievability oracle targets a *higher*
    # accuracy and is reported separately as accuracy headroom, not as the saving denominator).
    denom = p_cons
    for group in ("head_to_head_2stage", "incremental"):
        for d in cell[group].values():
            d["frac_of_oracle_consistency"] = (round(d["saving"] / denom, 3)
                                               if d["saving"] is not None and denom else None)
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="*", default=None)
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("-B", "--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-scale", action="store_true")
    ap.add_argument("--chain-perm-b", type=int, default=1000,
                    help="bootstrap draws for the chain-order-aware CI on scale cells")
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
            ts_conf = np.array([i / 10 for i in range(5, 11)])
            for m, rs in gk:
                sub = [r for r in rs if r["id"] in keep]
                kmax = len(sub[0]["samples"])
                cell = analyze_cell(
                    sub, m, kmax, args.k0, args.min_k, args.bootstrap, args.seed, config.GATE_DIR)
                # chain-order-aware CI for the headline confidence saving (fix: decoding-order variance)
                log(f"match-scale: chain-order bootstrap for {m} (B={args.chain_perm_b}) ...")
                cell["incremental"]["confidence"]["ci95_chainperm"] = _conf_chainperm_ci(
                    sub, kmax, args.min_k, ts_conf, args.chain_perm_b, args.seed)
                out["scale_matched"]["cells"][m] = cell

    dest = os.path.join(config.RESULTS_DIR, "analysis.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    log(f"analysis DONE -> {dest}")

    def fmt(d):
        s, (lo, hi) = d.get("saving"), d.get("ci95", [None, None])
        return "n.r." if s is None else f"{s:.3f} [{lo:.3f},{hi:.3f}]"
    print("\n=== head-to-head @ k0 (2-stage, matched mechanism) + incremental + yardsticks ===")
    print("| cell | n | full_acc | agree@k0 | trained@k0 | conf(inc) | esc | orc(cons/ach) | rand |")
    print("|---|--:|--:|---|---|---|---|---|--:|")
    for name, c in out["cells"].items():
        h, inc = c["head_to_head_2stage"], c["incremental"]
        r = c["random_stop"]["saving_at_iso_acc"]
        orc = f"{c['oracle_consistency']['saving']:.2f}/{c['oracle_achievable']['saving']:.2f}"
        print(f"| {name} | {c['n']} | {c['full_acc']} | {fmt(h.get('agreement',{}))} | "
              f"{fmt(h.get('trained',{}))} | {fmt(inc.get('confidence',{}))} | "
              f"{fmt(inc.get('esc',{}))} | {orc} | {'n.r.' if r is None else f'{r:.3f}'} |")
    print("\n=== paired Δ (trained − agreement), same problems: TOST @ margin 0.05 ===")
    print("| cell | Δ | 95% CI | TOST | MDE (½·CI) | censored |")
    print("|---|--:|---|---|--:|:--:|")
    for name, c in out["cells"].items():
        pd = c.get("paired_trained_minus_agreement")
        if not pd:
            continue
        d, (lo, hi) = pd["delta"], pd["ci95"]
        print(f"| {name} | {d:+.3f} | [{lo:+.3f},{hi:+.3f}] | {pd['tost']} | {pd['mde']:.3f} | "
              f"{'yes' if pd.get('censored') else 'no'} |")


if __name__ == "__main__":
    main()
