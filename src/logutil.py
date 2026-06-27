"""Timestamped, always-flushed progress logging so long GPU runs are never a black box.

Why this exists: a 3.5h Kaggle rollout with no heartbeat is impossible to reason about
("is it hung? which job? how far?"). Every long-running stage logs through here.
See CLAUDE.md §0.11 (LOGGING DISCIPLINE).

Each process gets its own elapsed clock (_T0 at import), which is exactly what we want:
the Kaggle runner spawns one `python -m scripts.run_rollouts` per job, so `+NNNs` is
"time into THIS job", and the wall clock disambiguates across jobs.
"""
import sys
import time
from datetime import datetime

_T0 = time.time()


def log(msg):
    """Print `[HH:MM:SS +elapsed] msg`, flushed immediately (Kaggle buffers stdout otherwise)."""
    el = time.time() - _T0
    wall = datetime.now().strftime("%H:%M:%S")
    print(f"[{wall} +{el:6.0f}s] {msg}", flush=True)


def fmt_eta(done, total, elapsed_s):
    """Human ETA string from progress so far. Returns e.g. '12.3 prob/min | ETA 8.4m'."""
    if elapsed_s <= 0 or done <= 0:
        return "rate n/a"
    rate = done / elapsed_s  # items per second
    remaining = max(0, total - done)
    eta_s = remaining / rate if rate > 0 else 0
    return f"{rate * 60:.1f}/min | {elapsed_s / 60:.1f}m elapsed | ETA {eta_s / 60:.1f}m"
