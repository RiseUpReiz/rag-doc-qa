"""LLM-as-judge grading of answers against a case's expected_behavior."""
import json
import re
import secrets
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from app.providers import get_judge_llm
from app.rag import extract_text
from evals.scoring import FAIL, PASS, is_daily_quota_error

JUDGE_ERROR = "JUDGE_ERROR"
PROMPT_PATH = Path(__file__).parent / "judge_prompt.txt"
QUOTA_REASON = "not judged: daily quota exhausted"

_FIELDS = ("question", "expected_behavior", "answer")
_PLACEHOLDER = re.compile(r"\{((?:question|expected_behavior|answer)(?:_tag)?)\}")
_FENCED = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)


@dataclass(frozen=True)
class Judgement:
    verdict: str
    reasoning: str


def build_prompt(question: str, expected_behavior: str, answer: str, token: str | None = None) -> str:
    """Fill the judge template in a single pass.

    Each call wraps the fields in tags with a random suffix (e.g. <answer_3f9a1c2e>), so an
    answer containing "</answer>" can't close its own tag and smuggle text outside it.
    One pass means text inside the question or answer that happens to look like a
    placeholder (e.g. "{expected_behavior}") is never substituted itself.
    """
    token = token or secrets.token_hex(4)
    values = {"question": question, "expected_behavior": expected_behavior, "answer": answer}
    values |= {f"{field}_tag": f"{field}_{token}" for field in _FIELDS}
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (``` or ```json), if present."""
    text = text.strip()
    match = _FENCED.match(text)
    return match.group(1).strip() if match else text


def parse_judgement(raw: str) -> Judgement:
    """Parse the judge's JSON reply. Anything unusable becomes JUDGE_ERROR with the raw text as reasoning."""
    try:
        data = json.loads(strip_fences(raw))
    except json.JSONDecodeError:
        return Judgement(JUDGE_ERROR, raw)
    if not isinstance(data, dict):
        return Judgement(JUDGE_ERROR, raw)

    verdict = data.get("verdict")
    verdict = verdict.strip().upper() if isinstance(verdict, str) else None
    if verdict not in (PASS, FAIL):
        return Judgement(JUDGE_ERROR, raw)

    reasoning = data.get("reasoning")
    return Judgement(verdict, reasoning if isinstance(reasoning, str) else "")


def judge_answer(question: str, expected_behavior: str, answer: str, llm=None) -> Judgement:
    """Ask the judge LLM whether an answer meets the expected behaviour."""
    if llm is None:
        llm = get_judge_llm()
    response = llm.invoke(build_prompt(question, expected_behavior, answer))
    return parse_judgement(extract_text(response))


def judge_all(items: Iterable[dict], delay: float = 0.0, llm=None, retries: int = 3) -> Iterator[Judgement]:
    """Judge items (dicts with question, expected_behavior, answer) in order, one Judgement each.

    A call that keeps failing yields JUDGE_ERROR rather than raising, so one bad call doesn't lose
    the rest of the run. Once the daily quota is exhausted, remaining items are not sent.
    """
    if llm is None:
        llm = get_judge_llm()
    quota_exhausted = False
    for i, item in enumerate(items):
        if quota_exhausted:
            yield Judgement(JUDGE_ERROR, QUOTA_REASON)
            continue
        if i and delay:
            time.sleep(delay)

        judgement = None
        for attempt in range(retries):
            try:
                judgement = judge_answer(item["question"], item["expected_behavior"], item["answer"], llm=llm)
                break
            except Exception as exc:
                if is_daily_quota_error(exc):
                    quota_exhausted = True
                if quota_exhausted or attempt == retries - 1:
                    judgement = Judgement(JUDGE_ERROR, f"error: {exc}")
                    break
                time.sleep(10 * (attempt + 1))
        yield judgement
