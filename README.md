# Adaptive Test-Time Compute

Do adaptive **self-consistency** stopping rules transfer across model scale and task? This repo
studies whether a stopping rule that decides *when to stop sampling* — calibrated on one model and
benchmark — still saves compute when moved to **smaller models** and **out-of-distribution tasks**.

It provides a small, reproducible harness to:
- generate self-consistency rollouts for open reasoning models,
- compare stopping methods head-to-head (fixed budget, ESC, confidence-threshold, a trained gate),
- measure the cost–accuracy trade-off and how it shifts across model size and benchmark.

## Install

```bash
pip install -r requirements.txt
```

Optional GPU fast path (CUDA only): `pip install vllm`.

## Quickstart

Offline pipeline check (CPU, no downloads):

```bash
python -m scripts.smoke_test
```

Real run (GPU recommended):

```bash
# 1) generate 16 samples/problem
python -m scripts.run_rollouts --dataset gsm8k --model qwen-1.5b --backend vllm --n 16 --limit 500

# 2) compute cost-accuracy for every method + plot
python -m scripts.run_eval --rollouts outputs/rollouts/gsm8k_qwen-1.5b.jsonl
```

## Datasets & models

- **Benchmarks:** GSM8K, MATH-500, BBH (GPQA-Diamond optional, gated). Downloaded via
  HuggingFace `datasets`; nothing is committed.
- **Models:** small open instruct models (Qwen2.5 0.5B/1.5B/7B, Llama-3.1-8B), all runnable on a
  single 16 GB GPU (quantize 7–8B).

## Project structure

```
src/        data, sampling, grader, gate, calibration, eval, plots
scripts/    smoke_test, run_rollouts, run_eval
kaggle/     remote GPU runner + CLI workflow
outputs/    rollouts, results, figures (git-ignored)
data/       benchmark cache (git-ignored)
```

## Running on Kaggle (free GPU)

See [`kaggle/KAGGLE_CLI.md`](kaggle/KAGGLE_CLI.md) for authoring locally and executing on Kaggle's
free GPU via the Kaggle API.

## License

MIT
