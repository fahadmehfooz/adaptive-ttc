"""Answer extraction + correctness. See CLAUDE.md section 1 and 5b (grading discipline).

GSM8K (number) + BBH (mc) grading is reliable. MATH (math) grading is approximate: latex
normalization + a sympy symbolic-equality fallback. Treat MATH accuracy as a lower bound, but
cross-method comparison at fixed grading is still valid.
"""
import re


def _last_boxed(text):
    """Return the content of the LAST \\boxed{...}, brace-matched (handles nesting). None if absent."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def extract_answer(text, kind):
    """Pull the final answer from a generation. Prefers \\boxed (math) then the 'Answer:' line."""
    if not text:
        return None
    if kind == "math":
        b = _last_boxed(text)
        if b is not None:
            return b
    m = re.findall(r"Answer:\s*(.+)", text)
    if m:
        return m[-1].strip()
    if kind == "math":
        b = _last_boxed(text)
        if b is not None:
            return b
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)  # last-resort: last number
    return nums[-1] if nums else None


def _math_norm(s):
    """Strip latex decoration to a comparable, sympy-friendly string."""
    s = s.strip()
    b = _last_boxed(s)
    if b is not None:
        s = b
    s = re.sub(r"\\text\s*\{(.*?)\}", r"\1", s)  # unwrap \text{X} -> X (keep text answers)
    s = s.replace("\\left", "").replace("\\right", "")
    for tok in ("\\!", "\\,", "\\;", "\\ ", "\\quad", "\\qquad", "$"):
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\pi", "pi")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace(" ", "").replace(",", "").rstrip(".")
    return s.lower()


def _math_equal(a, b):
    na, nb = _math_norm(a), _math_norm(b)
    if na == nb:
        return True
    try:
        import sympy
        ea = sympy.sympify(na.replace("^", "**"))
        eb = sympy.sympify(nb.replace("^", "**"))
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


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
    if pred is None:
        return False
    if kind == "math":
        return _math_equal(pred, gold)
    return normalize(pred, kind) == normalize(gold, kind)


def grade_samples(problem, texts):
    """Return list of {text, answer, correct} for each generation."""
    out = []
    for t in texts:
        a = extract_answer(t, problem.kind)
        out.append({"text": t, "answer": a, "correct": is_correct(a, problem.gold, problem.kind)})
    return out
