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
STAGE = "rollouts"           # "gpucheck" | "smoke" | "rollouts" | "eval"
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

    # Do NOT `pip install -r requirements.txt` on Kaggle — it perturbs the preinstalled stack.
    # Kaggle preinstalls torch/transformers/datasets/numpy/pandas/scikit-learn/joblib/matplotlib.
    #
    # BUT: Kaggle's GPU kernels are assigned a Tesla P100 (sm_60), and the stock torch
    # (2.10+cu128) dropped Pascal — arch_list is sm_70+ only -> cudaErrorNoKernelImageForDevice.
    # For the hf backend we install a cu121 torch build that still includes sm_60 kernels.
    if STAGE == "rollouts":
        if "vllm" in ARGS:
            sh("pip install -q vllm")  # NOTE: vLLM also dropped Pascal; needs a T4, not P100.
        else:
            sh("pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")

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
    elif STAGE == "eval":
        sh(f"python -m scripts.run_eval {ARGS}")
    else:
        print(f"unknown STAGE: {STAGE}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
