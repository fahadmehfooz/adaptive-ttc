"""Simulate fixed-budget and adaptive policies from precomputed rollouts.

Key trick (see CLAUDE.md S4): generate K=16 samples ONCE, then any budget/gate policy
is simulated offline at zero extra GPU cost.
"""
import json
from . import gate


def load_rollouts(path):
    """Each line: {id, dataset, gold, kind, samples:[{text, answer, correct}, ...]}."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _majority_correct(samples):
    """Is the majority answer over `samples` correct? Uses per-sample 'correct' flags."""
    ans, _ = gate.majority([s["answer"] for s in samples])
    if ans is None:
        return False
    for s in samples:
        if s["answer"] == ans:
            return bool(s["correct"])
    return False


def fixed_budget(rows, k):
    """Accuracy of majority-vote over first k samples. cost == k."""
    acc = sum(_majority_correct(r["samples"][:k]) for r in rows) / len(rows)
    return {"policy": f"fixed@{k}", "mean_cost": float(k), "accuracy": acc}


def _incremental(samples, kmax, should_stop, min_k):
    """Draw samples one at a time up to kmax; stop when should_stop(drawn) is True
    (only checked once at least min_k drawn). Returns (majority_correct, n_drawn)."""
    drawn = []
    for s in samples[:kmax]:
        drawn.append(s)
        if len(drawn) >= min_k and should_stop(drawn):
            break
    return _majority_correct(drawn), len(drawn)


def _esc_stop(drawn, window):
    """ESC rule: stop when the last `window` answers are all identical (and non-None)."""
    if len(drawn) < window:
        return False
    last = [d["answer"] for d in drawn[-window:]]
    return last[0] is not None and all(a == last[0] for a in last)


def esc_baseline(rows, window=4, kmax=16):
    """Early-Stopping Self-Consistency baseline (window-of-4 agreement). See lit_scan.md."""
    correct = cost = 0
    for r in rows:
        c, k = _incremental(r["samples"], kmax, lambda d: _esc_stop(d, window), min_k=window)
        correct += c
        cost += k
    n = len(rows)
    return {"policy": "esc", "window": window, "mean_cost": cost / n, "accuracy": correct / n}


def confidence_sweep(rows, kmax=16, min_k=2, thresholds=None):
    """Incremental confidence/agreement-threshold stop (ASC/Self-Calibration-style baseline)."""
    if thresholds is None:
        thresholds = [i / 10 for i in range(5, 11)]  # 0.5 .. 1.0
    pts = []
    for t in thresholds:
        correct = cost = 0
        for r in rows:
            c, k = _incremental(r["samples"], kmax,
                                lambda d, t=t: gate.features(d)["agreement"] >= t, min_k=min_k)
            correct += c
            cost += k
        n = len(rows)
        pts.append({"policy": "adaptive-confidence", "threshold": t,
                    "mean_cost": cost / n, "accuracy": correct / n})
    return pts


def adaptive(rows, decide, k0, kmax):
    """Adaptive policy. `decide(first_k0_samples) -> bool (stop)`.
    Stop -> answer from first k0 (cost k0); else answer from kmax (cost kmax)."""
    correct, cost = 0, 0
    for r in rows:
        s = r["samples"]
        head = s[:k0]
        if decide(head):
            correct += _majority_correct(head)
            cost += k0
        else:
            correct += _majority_correct(s[:kmax])
            cost += kmax
    n = len(rows)
    return {"mean_cost": cost / n, "accuracy": correct / n}


def agreement_sweep(rows, k0=4, kmax=16, thresholds=None):
    """Sweep the agreement threshold -> a cost-accuracy curve."""
    if thresholds is None:
        thresholds = [i / 10 for i in range(2, 11)]  # 0.2 .. 1.0
    pts = []
    for t in thresholds:
        res = adaptive(rows, lambda head, t=t: gate.agreement_stop(head, t), k0, kmax)
        pts.append({"policy": "adaptive-agreement", "threshold": t, **res})
    return pts


def baselines_and_adaptive(rows, k0=4, kmax=16):
    """Full head-to-head suite (S4): fixed budgets, ESC, incremental confidence, and the
    2-stage agreement policy — as one list of dicts."""
    out = [fixed_budget(rows, k) for k in (1, 4, 8, kmax)]
    out.append(esc_baseline(rows, window=4, kmax=kmax))
    out += confidence_sweep(rows, kmax=kmax)
    out += agreement_sweep(rows, k0=k0, kmax=kmax)
    return out
