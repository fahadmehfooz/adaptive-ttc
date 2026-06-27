"""S6 (headline #2): does a stopping rule calibrated on ONE dataset transfer to others?

Trains the gate on a source rollouts file, then evaluates every method (fixed/ESC/confidence/
agreement/trained) on each target rollouts file. Emits a transfer table: for each (method, target)
the compute saved at iso-accuracy vs the full-budget baseline.

Run:
  python -m scripts.transfer_matrix \
      --train outputs/rollouts/gsm8k_qwen-1.5b.jsonl \
      --targets outputs/rollouts/gsm8k_qwen-1.5b.jsonl outputs/rollouts/math500_qwen-1.5b.jsonl \
      --k0 4
"""
import argparse
import json
import os

from src import config, eval as ev, gate as gatemod, calibration
from src.logutil import log
from scripts.train_gate import build_training


def full_budget_acc(rows, kmax):
    return ev.fixed_budget(rows, kmax)["accuracy"]


def best_cost_at_acc(points, target_acc, tol=0.01):
    """Cheapest policy point that reaches (target_acc - tol). None if none qualifies."""
    ok = [p for p in points if p["accuracy"] >= target_acc - tol]
    return min((p["mean_cost"] for p in ok), default=None)


def summarize(rows, gate_model, k0, kmax):
    """All methods on one target -> list of policy points."""
    pts = ev.baselines_and_adaptive(rows, k0=k0, kmax=kmax)
    if gate_model is not None:
        pts += ev.trained_gate_sweep(rows, gate_model, k0=k0, kmax=kmax)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="source rollouts to train the gate on")
    ap.add_argument("--targets", nargs="+", required=True, help="target rollouts to evaluate on")
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=config.SAMPLING["n"])
    args = ap.parse_args()

    config.ensure_dirs()

    log(f"transfer_matrix START train={os.path.basename(args.train)} "
        f"targets={[os.path.basename(t) for t in args.targets]} k0={args.k0} kmax={args.kmax}")

    # Train gate on source.
    log(f"loading + training gate on source {os.path.basename(args.train)} ...")
    train_rows = ev.load_rollouts(args.train)
    feats, labels = build_training(train_rows, args.k0)
    gate_model = None
    if len(set(labels)) >= 2:
        gate_model = gatemod.TrainedGate().fit(feats, labels)
        confs = [gate_model.predict_proba(f) for f in feats]
        log(f"gate trained on {os.path.basename(args.train)} "
            f"({len(train_rows)} rows) | train ECE {calibration.ece(confs, labels):.4f}")
    else:
        log("WARNING: source has one class — skipping trained-gate row.")

    # Evaluate every method on every target; collect iso-accuracy compute savings.
    matrix = {}
    for ti, tgt in enumerate(args.targets, 1):
        log(f"[target {ti}/{len(args.targets)}] evaluating {os.path.basename(tgt)} ...")
        rows = ev.load_rollouts(tgt)
        kmax = args.kmax
        full_acc = full_budget_acc(rows, kmax)
        pts = summarize(rows, gate_model, args.k0, kmax)
        log(f"[target {ti}/{len(args.targets)}] {os.path.basename(tgt)}: "
            f"{len(rows)} rows, full_acc={full_acc:.3f}, {len(pts)} policy points")
        per_method = {}
        for method in sorted({p["policy"] for p in pts}):
            mpts = [p for p in pts if p["policy"] == method]
            cost = best_cost_at_acc(mpts, full_acc)
            per_method[method] = {
                "cost_at_full_acc": cost,
                "savings_vs_full": (None if cost is None else round(1 - cost / kmax, 3)),
            }
        matrix[os.path.basename(tgt)] = {"full_acc": round(full_acc, 3), "methods": per_method}

    out = os.path.join(config.RESULTS_DIR, "transfer_matrix.json")
    with open(out, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\ntransfer matrix -> {out}\n")
    # Pretty print: rows = targets, value = trained-gate savings (headline)
    for tgt, d in matrix.items():
        tg = d["methods"].get("adaptive-trained", {})
        print(f"  {tgt:38} full_acc={d['full_acc']:.3f}  trained-gate savings={tg.get('savings_vs_full')}")


if __name__ == "__main__":
    main()
