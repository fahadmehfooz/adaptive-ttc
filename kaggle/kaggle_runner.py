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
import time
from datetime import datetime

# ---- EDIT THESE PER RUN -----------------------------------------------------
REPO_URL = "https://github.com/fahadmehfooz/adaptive-ttc.git"
STAGE = "rollouts"  # "gpucheck" | "smoke" | "rollouts" | "rollouts_many" | "eval"
# 7B scale endpoint via T4×2 + vLLM (user toggled accelerator). VALIDATION FIRST (limit 8): the
# GPU-identity log line confirms we actually got 2× T4 (not P100), and tests the vLLM stack, before
# the full 500-problem run. tensor_parallel auto = #GPUs. (4-bit P100 fallback: --backend hf
# --quantization 4bit --gen-batch 4, kept working in HFSampler.)
ARGS = "--dataset gsm8k --model qwen-7b --backend vllm --n 8 --limit 8"

# For STAGE="rollouts_many": run several rollouts in ONE GPU session (one torch install,
# models cached within the session) — far cheaper on quota than one kernel per config.
# v12: the two remaining 0.5B transfer rollouts (small/fast, safely under the runtime cap).
JOBS = [
    "--dataset bbh     --model qwen-0.5b --backend hf --n 16 --limit 200",
    "--dataset math500 --model qwen-0.5b --backend hf --n 16 --limit 200",
]
# -----------------------------------------------------------------------------

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "repo")


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def sh(cmd):
    print(f"[{_ts()}] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    if not os.path.exists(REPO):
        sh(f"git clone --depth 1 {REPO_URL} {REPO}")
    os.chdir(REPO)

    # Point outputs/ at the persisted Kaggle working dir.
    sh(f"rm -rf {REPO}/outputs && ln -s {WORK}/outputs {REPO}/outputs && mkdir -p {WORK}/outputs")

    # Do NOT `pip install -r requirements.txt` on Kaggle — it perturbs the preinstalled stack.
    # Kaggle preinstalls torch/transformers/datasets/numpy/pandas/scikit-learn/joblib/matplotlib.
    #
    # BUT: Kaggle's GPU kernels are assigned a Tesla P100 (sm_60), and the stock torch
    # (2.10+cu128) dropped Pascal — arch_list is sm_70+ only -> cudaErrorNoKernelImageForDevice.
    # For the hf backend we install a cu121 torch build that still includes sm_60 kernels.
    # Install matching torch+torchvision from cu121 (sm_60 P100 support; torchvision must match
    # torch's ABI or `torchvision::nms does not exist` breaks transformers' model import).
    def install_p100_torch():
        sh("pip install -q torch==2.5.1 torchvision==0.20.1 "
           "--index-url https://download.pytorch.org/whl/cu121")

    if STAGE in ("rollouts", "rollouts_many"):
        all_args = ARGS if STAGE == "rollouts" else " ".join(JOBS)
        if "vllm" in all_args:
            sh("pip install -q vllm")  # NOTE: vLLM also dropped Pascal; needs a T4, not P100.
        else:
            install_p100_torch()
            # 4-bit (7B/8B on 16GB P100) needs bitsandbytes; must match the cu121 torch above.
            if "4bit" in all_args or "nf4" in all_args:
                sh("pip install -q bitsandbytes")

    if STAGE == "gpucheck":
        sh('nvidia-smi || true')
        sh('python -c "'
           'import torch; '
           "print('torch', torch.__version__, '| cuda', torch.version.cuda); "
           "print('available', torch.cuda.is_available()); "
           "print('device', torch.cuda.get_device_name(0)); "
           "print('capability sm_', torch.cuda.get_device_capability(0)); "
           "print('arch_list', torch.cuda.get_arch_list()); "
           "x = torch.randn(4, device='cuda'); print('basic cuda op:', float((x+1).sum()))"
           '"')
    elif STAGE == "smoke":
        sh("python -m scripts.smoke_test")
    elif STAGE == "rollouts":
        sh(f"python -m scripts.run_rollouts {ARGS}")
    elif STAGE == "rollouts_many":
        # Resilient: a failing job (e.g. one dataset loader) must not kill the rest.
        t_all = time.time()
        for ji, j in enumerate(JOBS, 1):
            print(f"\n===== [{_ts()}] JOB {ji}/{len(JOBS)} START: {j} =====", flush=True)
            t0 = time.time()
            try:
                sh(f"python -m scripts.run_rollouts {j}")
                print(f"===== [{_ts()}] JOB {ji}/{len(JOBS)} DONE in "
                      f"{(time.time() - t0) / 60:.1f}m =====", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"!! [{_ts()}] JOB {ji}/{len(JOBS)} FAILED (continuing): {j}\n   {e}",
                      file=sys.stderr, flush=True)
        print(f"\n===== [{_ts()}] ALL {len(JOBS)} JOBS FINISHED in "
              f"{(time.time() - t_all) / 60:.1f}m =====", flush=True)
    elif STAGE == "eval":
        sh(f"python -m scripts.run_eval {ARGS}")
    else:
        print(f"unknown STAGE: {STAGE}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
