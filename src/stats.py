"""Bootstrap confidence intervals over problems (see CRITIQUE_LOG iter 1, fixes #2/#4).

The saving metric involves a data-dependent operating-point selection (cheapest threshold that
reaches full-budget accuracy), so we bootstrap the WHOLE procedure: resample problems with
replacement, recompute full-budget accuracy, re-select the operating point, recompute saving.
This captures both sampling and selection variance.
"""
import random


def bootstrap_ci(rows, stat_fn, B=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for stat_fn(rows_resample).

    Returns (point, lo, hi, n_valid). point = stat on the full sample; lo/hi = (alpha/2, 1-alpha/2)
    percentiles over B resamples. stat_fn may return None (e.g. no threshold reaches the bar);
    such draws are dropped and counted via n_valid.
    """
    point = stat_fn(rows)
    rng = random.Random(seed)
    n = len(rows)
    vals = []
    for _ in range(B):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        v = stat_fn(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return (point, None, None, 0)
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (point, lo, hi, len(vals))


def ci_overlap(a, b):
    """True if two (lo, hi) intervals overlap. None-safe (returns True if either is undefined —
    i.e. we cannot claim separation)."""
    (alo, ahi), (blo, bhi) = a, b
    if None in (alo, ahi, blo, bhi):
        return True
    return not (ahi < blo or bhi < alo)
