"""Paths, model registry, and sampling defaults. See CLAUDE.md sections 3 and 4."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(REPO_ROOT, "data")
HF_CACHE = os.path.join(DATA_DIR, "hf_cache")

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
ROLLOUTS_DIR = os.path.join(OUTPUT_DIR, "rollouts")
GATE_DIR = os.path.join(OUTPUT_DIR, "gate")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Model registry — keys used everywhere; HF ids resolved here only.
MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
}
DEFAULT_MODEL = "qwen-1.5b"

# Self-consistency sampling defaults.
SAMPLING = dict(n=16, temperature=0.8, top_p=0.95, max_new_tokens=1024)


def ensure_dirs():
    for d in (DATA_DIR, HF_CACHE, OUTPUT_DIR, ROLLOUTS_DIR, GATE_DIR, RESULTS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)
