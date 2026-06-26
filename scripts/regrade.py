"""Re-grade a rollout file IN PLACE using the current grader, on the stored generation text.

Lets us improve answer extraction / grading without re-running the GPU (the raw `text` of every
sample is kept in the rollout). Rewrites each sample's `answer` and `correct`.

Run:
  python -m scripts.regrade --rollouts outputs/rollouts/gsm8k_qwen-1.5b.jsonl
"""
import argparse
import json

from src import data, grader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    changed = 0
    for r in rows:
        # Reconstruct a minimal Problem just to carry gold + kind.
        p = data.Problem(r["id"], r["dataset"], "", r["gold"], r["kind"])
        regraded = grader.grade_samples(p, [s["text"] for s in r["samples"]])
        for old, new in zip(r["samples"], regraded):
            if old.get("correct") != new["correct"] or old.get("answer") != new["answer"]:
                changed += 1
            old["answer"], old["correct"] = new["answer"], new["correct"]

    with open(args.rollouts, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    n = sum(len(r["samples"]) for r in rows)
    print(f"regraded {args.rollouts}: {changed}/{n} samples changed")


if __name__ == "__main__":
    main()
