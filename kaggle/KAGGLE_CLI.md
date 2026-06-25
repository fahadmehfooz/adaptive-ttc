# Kaggle as a remote GPU runner

You author code locally; Kaggle runs it on a free P100/2×T4 (30 GPU-h/week). The GPU
lives on Kaggle's servers — you cannot use it locally, but you never touch the browser UI.

## One-time setup
1. Install the CLI: `pip install kaggle`
2. Get an API token: kaggle.com → Account → **Create New API Token** → downloads `kaggle.json`.
3. Place it: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
4. In `kaggle/kernel-metadata.json`, set `id` to `<your-kaggle-username>/adaptive-ttc-runner`.
5. In `kaggle/kaggle_runner.py`, set `REPO_URL` to your GitHub repo (push this folder there first).

## The loop (run from the repo root)
```bash
# 1. Develop locally + smoke test (offline, CPU):
python -m scripts.smoke_test

# 2. Commit & push code to GitHub so the kernel can clone it:
git add -A && git commit -m "run X" && git push

# 3. Edit kaggle/kaggle_runner.py -> set STAGE and ARGS for this run.

# 4. Push the kernel (it runs headless on GPU). From the kaggle/ dir:
cd kaggle && kaggle kernels push

# 5. Watch status until complete:
kaggle kernels status <your-username>/adaptive-ttc-runner

# 6. Pull results back into ./outputs locally:
kaggle kernels output <your-username>/adaptive-ttc-runner -p ../outputs
```

## Notes
- **Sessions cap ~9–12h, 30h/week.** Keep runs checkpointed — `run_rollouts.py` already
  skips ids already in the JSONL, so re-pushing resumes instead of restarting.
- GPU + internet are enabled in `kernel-metadata.json`. Internet is needed to clone the
  repo and download HF datasets/models.
- For 7B/8B on a 16 GB card, pass `--quantization awq` (and use an AWQ model id).
- Alternative to GitHub clone: upload the repo as a **Kaggle Dataset** and attach it via
  `dataset_sources` in the metadata, then import from `/kaggle/input/...` instead of cloning.
