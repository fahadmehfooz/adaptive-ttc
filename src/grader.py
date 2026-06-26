"""Answer extraction + correctness. See CLAUDE.md section 1."""
import re


def extract_answer(text, kind):
    """Pull the final answer from a generation. Prefers the 'Answer:' line."""
    if not text:
        return None
    m = re.findall(r"Answer:\s*(.+)", text)
    raw = m[-1].strip() if m else None

    if raw is None and kind == "math":
        b = re.findall(r"\\boxed\{([^{}]*)\}", text)
        raw = b[-1] if b else None

    if raw is None:  # last-resort: last number in the text
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        raw = nums[-1] if nums else None

    return raw


def normalize(s, kind):
    if s is None:
        return None
    s = s.strip().strip(".").strip("$").strip("()").replace(",", "").replace(" ", "")
    if kind == "number":
        # GSM8K convention: the final answer is the LAST number stated.
        nums = re.findall(r"-?\d+\.?\d*", s)
        if not nums:
            return s
        v = nums[-1]
        return v.rstrip("0").rstrip(".") if "." in v else v
    return s.lower()


def is_correct(pred, gold, kind):
    return normalize(pred, kind) is not None and normalize(pred, kind) == normalize(gold, kind)


def grade_samples(problem, texts):
    """Return list of {text, answer, correct} for each generation."""
    out = []
    for t in texts:
        a = extract_answer(t, problem.kind)
        out.append({"text": t, "answer": a, "correct": is_correct(a, problem.gold, problem.kind)})
    return out
