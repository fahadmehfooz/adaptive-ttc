"""Thin Kaggle script-kernel runner. Runs on Kaggle's GPU; you author locally.

Workflow (see kaggle/KAGGLE_CLI.md):
  1. Push your code to GitHub (set REPO_URL below).
  2. Edit STAGE / ARGS below for the run you want.
  3. `kaggle kernels push` from the kaggle/ dir -> it runs headless on the GPU.
  4. `kaggle kernels output` to pull outputs/ back down locally.

Outputs are written under /kaggle/working/outputs/ which Kaggle saves automatically.
"""
import os
import subprocess
import sys

# ---- EDIT THESE PER RUN -----------------------------------------------------
REPO_URL = "https://github.com/fahadmehfooz/adaptive-ttc.git"
STAGE = "rollouts"            # "smoke" | "rollouts" | "eval"
ARGS = "--dataset gsm8k --model qwen-1.5b --backend hf --n 8 --limit 16"
# -----------------------------------------------------------------------------

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "repo")


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    if not os.path.exists(REPO):
        sh(f"git clone --depth 1 {REPO_URL} {REPO}")
    os.chdir(REPO)

    # Point outputs/ at the persisted Kaggle working dir.
    sh(f"rm -rf {REPO}/outputs && ln -s {WORK}/outputs {REPO}/outputs && mkdir -p {WORK}/outputs")

    # IMPORTANT: do NOT `pip install -r requirements.txt` on Kaggle. It upgrades the
    # CUDA-matched torch to a generic wheel and breaks the GPU with
    # `cudaErrorNoKernelImageForDevice`. Kaggle preinstalls torch/transformers/datasets/
    # numpy/pandas/scikit-learn/joblib/matplotlib — we rely on those as-is.
    if STAGE == "rollouts" and "vllm" in ARGS:  # vLLM only when that backend is selected
        sh("pip install -q vllm")

    if STAGE == "smoke":
        sh("python -m scripts.smoke_test")
    elif STAGE == "rollouts":
        sh(f"python -m scripts.run_rollouts {ARGS}")
    elif STAGE == "eval":
        sh(f"python -m scripts.run_eval {ARGS}")
    else:
        print(f"unknown STAGE: {STAGE}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
