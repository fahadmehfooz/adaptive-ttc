"""S2 smoke test: toy data -> FakeSampler -> grade -> rollouts -> eval -> plot.
Offline, CPU, no downloads. See CLAUDE.md step S2.

Run: python -m scripts.smoke_test
"""
import json
import os

from src import config, data, grader, sampling, eval as ev, plots


def main():
    config.ensure_dirs()
    problems = data.load_problems("toy")
    sampler = sampling.get_sampler("fake", correct_rate=0.7)

    raw = sampler.sample(problems, n=config.SAMPLING["n"])

    rollout_path = os.path.join(config.ROLLOUTS_DIR, "toy_fake.jsonl")
    with open(rollout_path, "w") as f:
        for p, texts in zip(problems, raw):
            graded = grader.grade_samples(p, texts)
            f.write(json.dumps({
                "id": p.id, "dataset": p.dataset, "gold": p.gold,
                "kind": p.kind, "samples": graded,
            }) + "\n")

    rows = ev.load_rollouts(rollout_path)
    points = ev.baselines_and_adaptive(rows, k0=4, kmax=config.SAMPLING["n"])

    results_path = os.path.join(config.RESULTS_DIR, "toy_fake.json")
    with open(results_path, "w") as f:
        json.dump(points, f, indent=2)

    fig_path = os.path.join(config.FIGURES_DIR, "toy_fake.png")
    plots.cost_accuracy(points, fig_path, title="Toy smoke test")

    print(f"OK  rollouts -> {rollout_path}")
    print(f"OK  results  -> {results_path}")
    print(f"OK  figure   -> {fig_path}")
    print("\nFixed budgets:")
    for p in points:
        if p["policy"].startswith("fixed"):
            print(f"  {p['policy']:>10}  cost={p['mean_cost']:.1f}  acc={p['accuracy']:.3f}")


if __name__ == "__main__":
    main()
