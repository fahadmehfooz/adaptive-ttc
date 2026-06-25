"""S4-S8: compute fixed-budget + adaptive results and the cost-accuracy figure.
See CLAUDE.md steps S4-S8.

Run:
  python -m scripts.run_eval --rollouts outputs/rollouts/gsm8k_qwen-1.5b.jsonl
"""
import argparse
import json
import os

from src import config, eval as ev, plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=config.SAMPLING["n"])
    args = ap.parse_args()

    config.ensure_dirs()
    rows = ev.load_rollouts(args.rollouts)
    points = ev.baselines_and_adaptive(rows, k0=args.k0, kmax=args.kmax)

    tag = os.path.splitext(os.path.basename(args.rollouts))[0]
    results_path = os.path.join(config.RESULTS_DIR, f"{tag}.json")
    with open(results_path, "w") as f:
        json.dump(points, f, indent=2)
    fig_path = os.path.join(config.FIGURES_DIR, f"{tag}.png")
    plots.cost_accuracy(points, fig_path, title=tag)

    print(f"results -> {results_path}")
    print(f"figure  -> {fig_path}")
    for p in points:
        line = f"  {p['policy']:>20}  cost={p['mean_cost']:.2f}  acc={p['accuracy']:.3f}"
        if "threshold" in p:
            line += f"  (t={p['threshold']})"
        print(line)


if __name__ == "__main__":
    main()
