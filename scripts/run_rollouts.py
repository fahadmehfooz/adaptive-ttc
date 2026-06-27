"""S3: generate rollouts for a (dataset, model). See CLAUDE.md step S3.

Run (GPU):
  python -m scripts.run_rollouts --dataset gsm8k --model qwen-1.5b --backend vllm --n 16 --limit 500
"""
import argparse
import json
import os
import time

from src import config, data, grader, sampling
from src.logutil import log, fmt_eta


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

    log(f"run_rollouts START dataset={args.dataset} model={args.model} "
        f"backend={args.backend} n={args.n} limit={args.limit} gen_batch={args.gen_batch}")

    config.ensure_dirs()

    # Log the GPU we actually got — confirms accelerator (P100 vs T4×2) before any long run.
    try:
        import torch
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            log(f"GPU: {torch.cuda.device_count()}× {names}")
        else:
            log("GPU: none (CPU)")
    except Exception as e:
        log(f"GPU: could not query ({e})")

    log(f"loading dataset {args.dataset} (split={args.split}, limit={args.limit}) ...")
    problems = data.load_problems(args.dataset, limit=args.limit, split=args.split)
    log(f"loaded {len(problems)} problems")

    log(f"initializing sampler backend={args.backend} model={args.model} "
        f"(this loads weights — slow for big models) ...")
    sampler = sampling.get_sampler(args.backend, args.model,
                                   quantization=args.quantization, gen_batch=args.gen_batch)
    log("sampler ready")

    out_path = os.path.join(config.ROLLOUTS_DIR, f"{args.dataset}_{args.model}.jsonl")

    # Resumable: skip ids already written (see CLAUDE.md S6 'checkpoint/resumable').
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    todo = [p for p in problems if p.id not in done]
    log(f"{len(done)} already done, {len(todo)} to generate -> {out_path}")

    BATCH = 32
    t_start = time.time()
    with open(out_path, "a") as f:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            log(f"chunk {i // BATCH + 1}/{(len(todo) + BATCH - 1) // BATCH}: "
                f"generating problems {i}..{min(i + BATCH, len(todo))} of {len(todo)} "
                f"(n={args.n} samples each)")
            raw = sampler.sample(chunk, n=args.n)
            for p, texts in zip(chunk, raw):
                graded = grader.grade_samples(p, texts)
                f.write(json.dumps({
                    "id": p.id, "dataset": p.dataset, "gold": p.gold,
                    "kind": p.kind, "samples": graded,
                }) + "\n")
            f.flush()
            done_n = min(i + BATCH, len(todo))
            log(f"saved {done_n}/{len(todo)} -> {out_path} | {fmt_eta(done_n, len(todo), time.time() - t_start)}")
    log(f"run_rollouts DONE dataset={args.dataset} model={args.model} "
        f"({len(todo)} new, {len(done)} pre-existing) in {(time.time() - t_start) / 60:.1f}m")


if __name__ == "__main__":
    main()
