"""Apply the ADAPTIVE-STOP audit (§3) to a PUBLISHED method's released rollouts — Adaptive-Consistency
(ASC, Aggarwal et al., EMNLP 2023). Demonstrates the audit on the field's most-cited baseline, not just
our own harness (CRITIQUE_LOG iter-4 blocker #2 → main-track lever).

ASC released per-question × 40-sample data (answers + 0/1 scores + target) for 2 models × 13 datasets ×
3 seeds:
  git clone https://github.com/Pranjal2041/AdaptiveConsistency && bash download_outputs.sh
  # or: gdown 11ripw7-E5T8a2TZUiD5vbC7QuR_qfZOA  (outputs.zip, ~72MB)
Point --asc-root at the unzipped `adaptive_consistency_outputs/`.

Guards applied (all CPU, from their samples):
  - reproduce ASC-style saving = 1 − mean_cost/K at iso-accuracy vs fixed SC@K (their baseline);
  - bootstrap 95% CI over problems (ASC's Table-1 headline is a point estimate);
  - oracle (consistency) upper bound — ASC never reports one;
  - random-stop control — how much saving is free;
  - calibration of the agreement signal: ECE, identity ratio ECE/|mean_conf−acc|, isotonic ECE
    (ASC never validates its Beta/Dirichlet stopping probability; its Limitations concedes the trigger
    can fire on unstable majorities);
  - answer-space cardinality (mean #distinct answers) — ties ASC's best cells to our §5.3.
"""
import argparse
import glob
import json
import os

import numpy as np

from src import gate, eval as ev
from src.calibration import ece
from src.logutil import log

K = 40
TOL = 0.01


def load_asc(path, limit=None):
    """ASC jsonl -> our rollout rows: {samples:[{answer, correct, text}]}."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            ans, sc = d["answers"], d["scores"]
            gens = d.get("generation", [""] * len(ans))

            def _hashable(a):
                if a is None:
                    return None
                if isinstance(a, (list, dict)):
                    return json.dumps(a, sort_keys=True)
                return a
            samples = [{"answer": _hashable(a),
                        "correct": bool(sc[j]),
                        "text": gens[j] if j < len(gens) else ""}
                       for j, a in enumerate(ans)]
            rows.append({"id": f"{os.path.basename(path)}-{i}", "samples": samples[:K]})
            if limit and len(rows) >= limit:
                break
    return rows


def _agr_incremental_arrays(rows, ts, min_k, kmax):
    n = len(rows)
    cost = np.zeros((len(ts), n))
    corr = np.zeros((len(ts), n))
    for ti, t in enumerate(ts):
        for i, r in enumerate(rows):
            c, k = ev._incremental(r["samples"], kmax,
                                   lambda d, t=t: gate.features(d)["agreement"] >= t, min_k=min_k)
            corr[ti, i] = c
            cost[ti, i] = k
    return cost, corr


def _boot_saving(cost, corr, full_correct, kmax, B, seed):
    rng = np.random.default_rng(seed)
    n = len(full_correct)

    def sav(idx):
        full = full_correct[idx].mean()
        accs = corr[:, idx].mean(axis=1)
        costs = cost[:, idx].mean(axis=1)
        ok = accs >= full - TOL
        return None if not ok.any() else 1 - costs[ok].min() / kmax

    pt = sav(np.arange(n))
    vals = [v for v in (sav(rng.integers(0, n, n)) for _ in range(B)) if v is not None]
    if not vals or pt is None:
        return (None, None, None)
    vals = np.sort(np.array(vals))
    return (round(float(pt), 3), round(float(vals[int(.025 * len(vals))]), 3),
            round(float(vals[min(len(vals) - 1, int(.975 * len(vals)))]), 3))


def _oracle_consistency(rows, kmax):
    cost = tot = 0
    for r in rows:
        s = r["samples"]
        final, _ = gate.majority([x["answer"] for x in s[:kmax]])
        k = kmax
        for j in range(1, kmax + 1):
            a, _ = gate.majority([x["answer"] for x in s[:j]])
            if a == final and a is not None:
                k = j
                break
        cost += k
        tot += 1
    return round(1 - (cost / tot) / kmax, 3)


def _random_stop(rows, kmax, full_acc, min_k=2, seed=0, reps=20):
    rng = np.random.default_rng(seed)
    best = None
    for p in [i / 20 for i in range(21)]:
        accs = costs = 0.0
        for _ in range(reps):
            cor = cst = 0
            for r in rows:
                s = r["samples"]
                if rng.random() < p:
                    cor += ev._majority_correct(s[:min_k]); cst += min_k
                else:
                    cor += ev._majority_correct(s[:kmax]); cst += kmax
            accs += cor / len(rows); costs += cst / len(rows)
        if accs / reps >= full_acc - TOL:
            sv = 1 - (costs / reps) / kmax
            best = sv if best is None else max(best, sv)
    return None if best is None else round(best, 3)


def _calibration(rows, kmax, seed=0):
    confs = [gate.features(r["samples"][:kmax])["agreement"] for r in rows]
    labels = [int(ev._majority_correct(r["samples"][:kmax])) for r in rows]
    n = len(rows)
    mc, ac = sum(confs) / n, sum(labels) / n
    e = ece(confs, labels)
    ratio = round(e / abs(mc - ac), 2) if abs(mc - ac) > 1e-9 else None
    # isotonic 2-fold
    from sklearn.isotonic import IsotonicRegression
    import random
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    h = n // 2; folds = [idx[:h], idx[h:]]; recal = [0.0] * n
    for fi in range(2):
        te, tr = folds[fi], folds[1 - fi]
        if len({labels[i] for i in tr}) < 2:
            for i in te: recal[i] = confs[i]
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit([confs[i] for i in tr], [labels[i] for i in tr])
        for i, pv in zip(te, ir.predict([confs[i] for i in te])): recal[i] = float(pv)
    return {"mean_conf": round(mc, 3), "acc": round(ac, 3), "ece": round(e, 3),
            "ece_over_gap": ratio, "ece_isotonic": round(ece(recal, labels), 3)}


def _cardinality(rows, kmax):
    ds = [len({x["answer"] for x in r["samples"][:kmax] if x["answer"] is not None}) for r in rows]
    return round(sum(ds) / len(ds), 2)


def audit_cell(rows, min_k, B, seed):
    kmax = min(K, len(rows[0]["samples"]))
    full_correct = np.array([ev._majority_correct(r["samples"][:kmax]) for r in rows], dtype=float)
    full_acc = float(full_correct.mean())
    ts = np.array([i / 20 for i in range(10, 21)])  # .5..1.0
    cost, corr = _agr_incremental_arrays(rows, ts, min_k, kmax)
    sv, lo, hi = _boot_saving(cost, corr, full_correct, kmax, B, seed)
    return {"n": len(rows), "kmax": kmax, "sc_full_acc": round(full_acc, 3),
            "asc_saving": sv, "asc_saving_ci95": [lo, hi],
            "oracle": _oracle_consistency(rows, kmax),
            "random_stop": _random_stop(rows, kmax, full_acc, min_k, seed),
            "mean_distinct_answers": _cardinality(rows, kmax),
            "calibration": _calibration(rows, kmax, seed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asc-root", required=True, help="unzipped adaptive_consistency_outputs/")
    ap.add_argument("--model", default="vicuna-13b")
    ap.add_argument("--datasets", nargs="+",
                    default=["boolean_expressions", "strategy_qa", "gsm", "svamp", "asdiv"])
    ap.add_argument("--seed-file", default="seed1")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("-B", "--bootstrap", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        pat = os.path.join(args.asc_root, args.model, ds, f"*{args.seed_file}*.jsonl")
        files = glob.glob(pat)
        if not files:
            log(f"SKIP {ds}: no file at {pat}")
            continue
        rows = load_asc(files[0], limit=args.limit)
        log(f"{ds}: {len(rows)} problems, auditing ...")
        results[ds] = audit_cell(rows, args.min_k, args.bootstrap, args.seed)  # noqa (seed set below)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        log(f"wrote {args.out}")

    print(f"\n=== ADAPTIVE-STOP audit of ASC ({args.model}, {args.seed_file}) ===")
    print("| dataset | n | SC acc | ASC saving [95% CI] | oracle | random | #distinct | ECE | ratio | ECE-iso |")
    print("|---|--:|--:|---|--:|--:|--:|--:|--:|--:|")
    for ds, r in results.items():
        c = r["calibration"]
        ci = r["asc_saving_ci95"]
        cis = "n.r." if r["asc_saving"] is None else f"{r['asc_saving']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        print(f"| {ds} | {r['n']} | {r['sc_full_acc']:.3f} | {cis} | {r['oracle']:.3f} | "
              f"{r['random_stop']} | {r['mean_distinct_answers']} | {c['ece']:.3f} | "
              f"{c['ece_over_gap']} | {c['ece_isotonic']:.3f} |")


if __name__ == "__main__":
    main()
