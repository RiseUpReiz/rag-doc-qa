import re
from dataclasses import dataclass, field

PASS, FAIL, ERROR = "PASS", "FAIL", "ERROR"

REFUSAL_MARKERS = (
    "don't know",
    "do not know",
    "not in the",
    "no information",
    "not mentioned",
    "not provided",
    "not specified",
    "cannot find",
    "can't find",
    "does not contain",
    "doesn't contain",
    "not available",
    "unable to answer",
    "cannot answer",
    "can't answer",
)


@dataclass
class Score:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def looks_like_refusal(answer: str) -> bool:
    text = answer.lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def score_case(case: dict, answer: str) -> Score:
    """Check one answer against the expectations declared in its test case."""
    text = answer.lower()
    reasons = []

    expected = case.get("must_contain_any")
    if expected and not any(term.lower() in text for term in expected):
        reasons.append(f"missing expected content (any of {expected})")

    for term in case.get("must_not_contain", []):
        if term.lower() in text:
            reasons.append(f"contains forbidden content: {term!r}")

    if case.get("expect_refusal") and not looks_like_refusal(answer):
        reasons.append("expected a refusal but got an answer")

    return Score(passed=not reasons, reasons=reasons)


def case_status(score: Score | None) -> str:
    """Classify a case: ERROR if the model call failed (no score), else PASS or FAIL."""
    if score is None:
        return ERROR
    return PASS if score.passed else FAIL


def pass_rate(statuses: list[str]) -> float | None:
    """Share of scored cases that passed. ERROR cases are excluded; None if nothing was scored."""
    scored = [s for s in statuses if s != ERROR]
    if not scored:
        return None
    return scored.count(PASS) / len(scored)


_QUOTA_ID = re.compile(r"""['"]quotaId['"]\s*:\s*['"]([^'"]+)['"]""")


def is_daily_quota_error(exc: BaseException) -> bool:
    """True if the exception (or one it was raised from) is a 429 for an exhausted per-day quota."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            if any("PerDay" in quota_id for quota_id in _QUOTA_ID.findall(text)):
                return True
        exc = exc.__cause__ or exc.__context__
    return False


def summarize_error(message: str, limit: int = 160) -> str:
    """Reduce an error message to one short line, dropping any trailing JSON payload."""
    line = message.strip().splitlines()[0] if message.strip() else ""
    line = line.split(" {", 1)[0]
    return line if len(line) <= limit else line[: limit - 1] + "…"


def overall_line(statuses: list[str]) -> str:
    """Headline result, e.g. '11/11 scored passed · 3 errors (not run)'."""
    errors = statuses.count(ERROR)
    scored = len(statuses) - errors
    return f"{statuses.count(PASS)}/{scored} scored passed · {errors} errors (not run)"
