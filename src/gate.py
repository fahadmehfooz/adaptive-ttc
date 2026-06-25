"""Gate logic: majority vote, distribution features, and gate variants.
See CLAUDE.md section 4 and section 5 (no hidden-state features yet)."""
import math
from collections import Counter


def majority(answers):
    """Return (majority_answer, agreement_fraction) ignoring None answers."""
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, 0.0
    c = Counter(answers)
    ans, cnt = c.most_common(1)[0]
    return ans, cnt / len(answers)


def features(samples):
    """Distribution features over the first-k samples (list of {answer, text} dicts)."""
    answers = [s["answer"] for s in samples if s["answer"] is not None]
    n = len(answers)
    if n == 0:
        return {"agreement": 0.0, "distinct": 0, "entropy": 0.0, "mean_len": 0.0, "n": 0}
    c = Counter(answers)
    probs = [v / n for v in c.values()]
    return {
        "agreement": max(probs),
        "distinct": len(c),
        "entropy": -sum(p * math.log(p + 1e-12) for p in probs),
        "mean_len": sum(len(s["text"]) for s in samples) / len(samples),
        "n": n,
    }


def agreement_stop(samples, threshold):
    """Stop if agreement among the given samples >= threshold."""
    return features(samples)["agreement"] >= threshold


FEATURE_KEYS = ["agreement", "distinct", "entropy", "mean_len"]


class TrainedGate:
    """Small logistic head: P(majority answer is correct | features). Stop if P >= threshold.
    Trained on (features of first-k samples) -> (majority-of-full-K correct)."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.fitted = False

    @staticmethod
    def _vec(feats):
        return [feats[k] for k in FEATURE_KEYS]

    def fit(self, feats_list, labels):
        X = [self._vec(f) for f in feats_list]
        self.clf.fit(X, labels)
        self.fitted = True
        return self

    def predict_proba(self, feats):
        return float(self.clf.predict_proba([self._vec(feats)])[0, 1])

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        import joblib
        g = cls()
        g.clf = joblib.load(path)
        g.fitted = True
        return g
