"""S3: generate rollouts for a (dataset, model). See CLAUDE.md step S3.

Run (GPU):
  python -m scripts.run_rollouts --dataset gsm8k --model qwen-1.5b --backend vllm --n 16 --limit 500
"""
import argparse
import json
import os

from src import config, data, grader, sampling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "fake"])
    ap.add_argument("--n", type=int, default=config.SAMPLING["n"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--quantization", default=None, help="e.g. awq for 7B/8B on 16GB")
    ap.add_argument("--gen-batch", type=int, default=16,
                    help="hf: sequences per forward pass; lower (e.g. 4) for 7B/8B to avoid OOM")
    args = ap.parse_args()

    config.ensure_dirs()
    problems = data.load_problems(args.dataset, limit=args.limit, split=args.split)
    sampler = sampling.get_sampler(args.backend, args.model,
                                   quantization=args.quantization, gen_batch=args.gen_batch)

    out_path = os.path.join(config.ROLLOUTS_DIR, f"{args.dataset}_{args.model}.jsonl")

    # Resumable: skip ids already written (see CLAUDE.md S6 'checkpoint/resumable').
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    todo = [p for p in problems if p.id not in done]
    print(f"{len(done)} already done, {len(todo)} to generate -> {out_path}")

    BATCH = 32
    with open(out_path, "a") as f:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            raw = sampler.sample(chunk, n=args.n)
            for p, texts in zip(chunk, raw):
                graded = grader.grade_samples(p, texts)
                f.write(json.dumps({
                    "id": p.id, "dataset": p.dataset, "gold": p.gold,
                    "kind": p.kind, "samples": graded,
                }) + "\n")
            f.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")


if __name__ == "__main__":
    main()
