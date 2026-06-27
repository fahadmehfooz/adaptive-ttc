"""Benchmark loading + prompt building. See CLAUDE.md section 3 (data) and section 5 (stubs)."""
import hashlib
from dataclasses import dataclass
from . import config
from .logutil import log

# Representative BBH task subset for the transfer eval (S6). Spans answer formats:
# multiple-choice letters, yes/no, words, counts. Override via load_problems(..., tasks=[...]).
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "logical_deduction_three_objects", "movie_recommendation",
    "navigate", "reasoning_about_colored_objects", "snarks", "sports_understanding",
]


@dataclass
class Problem:
    id: str
    dataset: str
    question: str
    gold: str
    kind: str  # "number" | "math" | "mc"


# Offline toy set for smoke tests — no network needed (dataset="toy").
TOY = [
    Problem("toy-0", "toy", "What is 12 + 30?", "42", "number"),
    Problem("toy-1", "toy", "A box has 3 rows of 4 apples. How many apples?", "12", "number"),
    Problem("toy-2", "toy", "What is 100 - 1?", "99", "number"),
    Problem("toy-3", "toy", "What is 7 * 8?", "56", "number"),
    Problem("toy-4", "toy", "Half of 50 is?", "25", "number"),
    Problem("toy-5", "toy", "What is 9 + 9 + 9?", "27", "number"),
]


def load_problems(dataset, limit=None, split="test", **kwargs):
    """Return a list[Problem]. Raw data cached under config.HF_CACHE.
    kwargs: bbh accepts tasks=[...] to override the default task subset."""
    if dataset == "toy":
        probs = list(TOY)

    elif dataset == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=config.HF_CACHE)
        probs = []
        for i, ex in enumerate(ds):
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            probs.append(Problem(f"gsm8k-{split}-{i}", "gsm8k", ex["question"], gold, "number"))

    elif dataset == "math500":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=config.HF_CACHE)
        probs = [Problem(f"math500-{i}", "math500", ex["problem"], str(ex["answer"]), "math")
                 for i, ex in enumerate(ds)]

    elif dataset == "bbh":
        probs = _load_bbh(limit, tasks=kwargs.get("tasks"))

    elif dataset == "gpqa":
        probs = _load_gpqa(limit)

    else:
        raise ValueError(f"unknown dataset: {dataset}")

    if limit is not None:
        probs = probs[:limit]
    return probs


def _load_bbh(limit, tasks=None):
    """BBH across a set of tasks, sampled evenly. Gold = exact target string (kind='mc')."""
    from datasets import load_dataset
    tasks = tasks or BBH_TASKS
    per = max(1, limit // len(tasks)) if limit else None
    log(f"BBH: loading {len(tasks)} task-configs (~{per} each) — each is a separate HF download")
    probs = []
    for ti, task in enumerate(tasks):
        log(f"BBH: [{ti + 1}/{len(tasks)}] loading task '{task}' ...")
        ds = load_dataset("lukaemon/bbh", task, split="test", cache_dir=config.HF_CACHE)
        before = len(probs)
        for i, ex in enumerate(ds):
            if per is not None and i >= per:
                break
            probs.append(Problem(f"bbh-{task}-{i}", "bbh",
                                 ex["input"].strip(), str(ex["target"]).strip(), "mc"))
        log(f"BBH: [{ti + 1}/{len(tasks)}] task '{task}' -> {len(probs) - before} problems "
            f"(running total {len(probs)})")
    return probs


def _det_order(seed_key, n):
    """Deterministic permutation of range(n) keyed by a string — no RNG, stable across runs."""
    return sorted(range(n), key=lambda j: hashlib.md5(f"{seed_key}-{j}".encode()).hexdigest())


def _load_gpqa(limit):
    """GPQA-Diamond as 4-way MC with deterministically shuffled options. Gold = letter A-D.
    NOTE: gated dataset — requires `huggingface-cli login` + accepting the dataset terms."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=config.HF_CACHE)
    probs = []
    for i, ex in enumerate(ds):
        opts = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        order = _det_order(i, 4)            # where each original option lands
        gold = "ABCD"[order.index(0)]       # position of the correct answer (index 0)
        lines = [f"({'ABCD'[k]}) {opts[order[k]]}" for k in range(4)]
        q = ex["Question"].strip() + "\n" + "\n".join(lines)
        probs.append(Problem(f"gpqa-{i}", "gpqa", q, gold, "mc"))
    return probs


def build_prompt(p: Problem) -> str:
    """Zero-shot CoT. Model must end with a line 'Answer: <answer>' for the grader."""
    return (
        "Solve the problem. Reason step by step, then end with a single final line "
        "exactly in the form 'Answer: <answer>'.\n\n"
        f"Problem: {p.question}\n"
    )
