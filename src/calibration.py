"""Calibration utilities: expected calibration error + temperature scaling.
See CLAUDE.md step S6."""
import math


def ece(confidences, correct, n_bins=10):
    """Expected Calibration Error. confidences in [0,1], correct in {0,1}."""
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correct):
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, y))
    total = len(confidences) or 1
    err = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        err += (len(b) / total) * abs(conf - acc)
    return err


def temperature_scale(logits, correct, grid=None):
    """Pick a scalar temperature T minimizing NLL of sigmoid(logit/T). Returns best T.
    `logits` are pre-sigmoid scores; `correct` in {0,1}."""
    if grid is None:
        grid = [0.5 + 0.1 * i for i in range(26)]  # 0.5 .. 3.0

    def nll(T):
        s = 0.0
        for z, y in zip(logits, correct):
            p = 1.0 / (1.0 + math.exp(-z / T))
            p = min(max(p, 1e-9), 1 - 1e-9)
            s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return s

    return min(grid, key=nll)
