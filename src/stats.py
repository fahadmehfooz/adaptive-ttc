"""Paired-difference bootstrap, equivalence (TOST), and power helpers.

CRITIQUE_LOG iter 2 #1: two adaptive signals are evaluated on the SAME problems, so their
difference must be tested with a PAIRED bootstrap (identical resample indices), not by asking
whether two marginal CIs overlap. And "we found a tie" must be an equivalence claim (TOST against a
margin) or an explicit power/MDE statement — not merely "failure to reject".
"""
import numpy as np


def paired_delta(saving_fn_a, saving_fn_b, n, B=2000, seed=0, margin=0.05):
    """Paired bootstrap of Δ = b − a where each saving_fn maps an index array -> saving (or None).

    Returns dict: point Δ, 95% CI, 90% CI (for TOST), TOST verdict vs ±margin, and the achieved
    half-width (an empirical minimum-detectable-effect for this cell/n). Draws that yield None for
    either method are dropped (both use the SAME indices, so pairing is preserved).
    """
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    pa, pb = saving_fn_a(full), saving_fn_b(full)
    point = None if (pa is None or pb is None) else pb - pa
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        a, b = saving_fn_a(idx), saving_fn_b(idx)
        if a is not None and b is not None:
            deltas.append(b - a)
    if not deltas:
        return {"delta": point, "ci95": [None, None], "ci90": [None, None],
                "tost": "undefined", "mde": None, "boot_valid": 0}
    d = np.sort(np.array(deltas))
    lo95, hi95 = float(d[int(0.025 * len(d))]), float(d[min(len(d) - 1, int(0.975 * len(d)))])
    lo90, hi90 = float(d[int(0.05 * len(d))]), float(d[min(len(d) - 1, int(0.95 * len(d)))])
    # TOST: equivalent iff the 90% CI lies entirely within (-margin, +margin)
    if lo90 > -margin and hi90 < margin:
        verdict = "equivalent"
    elif lo95 > 0 or hi95 < 0:
        verdict = "different"          # 95% CI excludes 0
    else:
        verdict = "inconclusive"       # neither equivalent nor different -> underpowered
    return {"delta": None if point is None else round(point, 4),
            "ci95": [round(lo95, 4), round(hi95, 4)],
            "ci90": [round(lo90, 4), round(hi90, 4)],
            "tost_margin": margin, "tost": verdict,
            "mde": round((hi95 - lo95) / 2, 4), "boot_valid": len(deltas)}
