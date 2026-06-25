"""S6/S7: train a gate that predicts whether the majority answer (over full K samples) is
correct, from features of just the first k0 samples. Saves the gate + reports calibration (ECE).

The gate is the *trained* baseline comparator (see CLAUDE.md S4). Train it on one dataset's
rollouts (e.g. GSM8K) and transfer it to others in S6.

Run:
  python -m scripts.train_gate --rollouts outputs/rollouts/gsm8k_qwen-1.5b.jsonl --k0 4 --name gsm8k_qwen-1.5b
"""
import argparse
import os

from src import config, eval as ev, gate as gatemod, calibration


def build_training(rows, k0):
    """Features of first-k0 samples -> label = (majority over ALL samples is correct)."""
    feats, labels = [], []
    for r in rows:
        s = r["samples"]
        feats.append(gatemod.features(s[:k0]))
        labels.append(int(ev._majority_correct(s)))
    return feats, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--name", default=None, help="output basename for the saved gate")
    args = ap.parse_args()

    config.ensure_dirs()
    rows = ev.load_rollouts(args.rollouts)
    feats, labels = build_training(rows, args.k0)

    pos = sum(labels)
    print(f"{len(labels)} examples | positives (majority-correct): {pos} | negatives: {len(labels) - pos}")
    if len(set(labels)) < 2:
        print("WARNING: only one class present — cannot train a gate. Need rollouts with a mix "
              "of correct/incorrect majorities (real data will have this; toy_fake may not).")
        return

    gate = gatemod.TrainedGate().fit(feats, labels)

    # Calibration report on the training set (real eval uses held-out transfer in S6).
    confs = [gate.predict_proba(f) for f in feats]
    ece = calibration.ece(confs, labels)
    print(f"train ECE: {ece:.4f}  | mean predicted P(correct): {sum(confs)/len(confs):.3f}")

    name = args.name or os.path.splitext(os.path.basename(args.rollouts))[0]
    out = os.path.join(config.GATE_DIR, f"{name}.joblib")
    gate.save(out)
    print(f"saved gate -> {out}")


if __name__ == "__main__":
    main()
