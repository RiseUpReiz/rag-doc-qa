"""Measure how well the LLM judge agrees with human labels.

Usage (from the repo root):
    python -m evals.validate_judge
    python -m evals.validate_judge --delay 6

Runs the judge over evals/judge_validation/labels.jsonl and adversarial.jsonl, skipping
items with no human_label, and writes a report. False passes (human FAIL, judge PASS)
are listed first: they are the costly error, since they let a bad answer through.
"""
import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import settings
from evals.judge import JUDGE_ERROR, judge_all
from evals.scoring import FAIL, PASS, summarize_error

VALIDATION_DIR = Path(__file__).parent / "judge_validation"
DEFAULT_INPUTS = [VALIDATION_DIR / "labels.jsonl", VALIDATION_DIR / "adversarial.jsonl"]
JUDGE_COLUMNS = (PASS, FAIL, JUDGE_ERROR)


@dataclass
class Metrics:
    judged: int
    agreed: int
    confusion: Counter
    false_passes: list[dict]
    false_fails: list[dict]
    judge_errors: list[dict]

    @property
    def agreement_rate(self) -> float | None:
        return self.agreed / self.judged if self.judged else None


def load_items(paths: list[Path]) -> list[dict]:
    items = []
    for path in paths:
        if not path.exists():
            print(f"Skipping {path}: not found")
            continue
        with path.open(encoding="utf-8") as f:
            items += [json.loads(line) for line in f if line.strip()]
    return items


def select_labelled(items: list[dict]) -> tuple[list[dict], Counter]:
    """Keep items the judge can be checked on. Returns them plus a count of skipped items by reason."""
    selected, skipped = [], Counter()
    for item in items:
        label = item.get("human_label")
        if label is None:
            skipped["no human_label"] += 1
            continue
        if not item.get("expected_behavior"):
            skipped["no expected_behavior"] += 1
            continue
        label = str(label).strip().upper()
        if label not in (PASS, FAIL):
            raise ValueError(f"{item_ref(item)}: human_label must be PASS, FAIL or null, got {item['human_label']!r}")
        selected.append({**item, "human_label": label})
    return selected, skipped


def compute_metrics(rows: list[dict]) -> Metrics:
    """Compare human_label with judge_verdict. Agreement is over items the judge gave a verdict for."""
    confusion = Counter((r["human_label"], r["judge_verdict"]) for r in rows)
    judged = [r for r in rows if r["judge_verdict"] in (PASS, FAIL)]
    return Metrics(
        judged=len(judged),
        agreed=sum(1 for r in judged if r["human_label"] == r["judge_verdict"]),
        confusion=confusion,
        false_passes=[r for r in judged if r["human_label"] == FAIL and r["judge_verdict"] == PASS],
        false_fails=[r for r in judged if r["human_label"] == PASS and r["judge_verdict"] == FAIL],
        judge_errors=[r for r in rows if r["judge_verdict"] == JUDGE_ERROR],
    )


def item_ref(item: dict) -> str:
    return f"{item.get('source', '?')}:{item.get('id', '?')}"


def _disagreement(r: dict) -> list[str]:
    lines = [
        f"### {item_ref(r)}: human {r['human_label']}, judge {r['judge_verdict']}",
        f"**Question:** {r['question']}",
        "",
        f"**Expected behaviour:** {r['expected_behavior']}",
        "",
        f"**Answer:** {r['answer']}",
        "",
        f"**Judge reasoning:** {r['judge_reasoning']}",
        "",
    ]
    if r.get("note"):
        lines += [f"**Labeller note:** {r['note']}", ""]
    return lines


def render_report(metrics: Metrics, skipped: Counter, judge_model: str) -> str:
    rate = "—" if metrics.agreement_rate is None else f"{metrics.agreement_rate:.0%}"
    skipped_text = ", ".join(f"{n} {reason}" for reason, n in sorted(skipped.items())) or "none"
    same_model = " (same as the model under test)" if judge_model == settings.llm_model else ""
    lines = [
        f"# Judge validation — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Judge: `{judge_model}`{same_model}",
        "",
        f"**False passes (human FAIL, judge PASS): {len(metrics.false_passes)}**",
        "",
        f"**Agreement: {metrics.agreed}/{metrics.judged} ({rate})** of items the judge returned a verdict for"
        f" · {len(metrics.judge_errors)} judge errors · skipped: {skipped_text}",
        "",
        "## Confusion matrix",
        "",
        "| Human \\ Judge | " + " | ".join(JUDGE_COLUMNS) + " |",
        "|---|" + "---|" * len(JUDGE_COLUMNS),
    ]
    for human in (PASS, FAIL):
        counts = " | ".join(str(metrics.confusion[(human, judge)]) for judge in JUDGE_COLUMNS)
        lines.append(f"| {human} | {counts} |")

    lines += ["", "## False passes (human FAIL, judge PASS)", ""]
    if not metrics.false_passes:
        lines.append("None.")
    for r in metrics.false_passes:
        lines += _disagreement(r)

    lines += ["", "## False fails (human PASS, judge FAIL)", ""]
    if not metrics.false_fails:
        lines.append("None.")
    for r in metrics.false_fails:
        lines += _disagreement(r)

    if metrics.judge_errors:
        lines += ["", "## Judge errors", ""]
        for r in metrics.judge_errors:
            lines.append(f"- **{item_ref(r)}** (human {r['human_label']}): {summarize_error(r['judge_reasoning'])}")
    return "\n".join(lines)


def run_judge(items: list[dict], delay: float, llm=None) -> list[dict]:
    rows = []
    for item, judgement in zip(items, judge_all(items, delay=delay, llm=llm)):
        print(f"judge {judgement.verdict:<11} human {item['human_label']:<4} {item_ref(item)}")
        rows.append({**item, "judge_verdict": judgement.verdict, "judge_reasoning": judgement.reasoning})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="*", default=DEFAULT_INPUTS,
                        help="labelled jsonl files (default: labels.jsonl and adversarial.jsonl)")
    parser.add_argument("--out", type=Path, default=VALIDATION_DIR)
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between calls (free-tier rate limits)")
    args = parser.parse_args()

    if not settings.judge_model:
        parser.error("JUDGE_MODEL is not set in .env")

    items, skipped = select_labelled(load_items(args.inputs))
    if not items:
        parser.error("no labelled items to validate against")

    metrics = compute_metrics(run_judge(items, delay=args.delay))
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / f"report-{datetime.now():%Y%m%d-%H%M%S}.md"
    report_path.write_text(render_report(metrics, skipped, settings.judge_model), encoding="utf-8")

    rate = "—" if metrics.agreement_rate is None else f"{metrics.agreement_rate:.0%}"
    print(f"\nFalse passes: {len(metrics.false_passes)} · agreement {metrics.agreed}/{metrics.judged} ({rate})."
          f" Report: {report_path}")


if __name__ == "__main__":
    main()
