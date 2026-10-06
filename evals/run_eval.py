"""Run the safety and quality eval suite against the RAG pipeline.

Usage (from the repo root):
    python -m evals.run_eval
    python -m evals.run_eval --delay 6 --only indirect_injection
    python -m evals.run_eval --cases evals/attacks/cases.jsonl --corpus evals/attacks
    python -m evals.run_eval --rescore evals/reports/results-<stamp>.jsonl
    python -m evals.run_eval --cases evals/attacks/cases.jsonl --corpus evals/attacks --judge
    python -m evals.run_eval --rescore evals/reports/results-<stamp>.jsonl --judge --cases evals/attacks/cases.jsonl

--judge also grades each answer with the LLM judge (JUDGE_MODEL) against the case's
expected_behavior. With --rescore, --cases fills in expected_behavior for saved results
that predate it.
"""
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma

from app.config import settings
from app.ingest import chunk_documents, load_documents
from app.providers import get_embeddings
from app.rag import answer_question
from evals.judge import JUDGE_ERROR, judge_all
from evals.scoring import (
    ERROR,
    FAIL,
    PASS,
    case_status,
    is_daily_quota_error,
    overall_line,
    pass_rate,
    score_case,
    summarize_error,
)

EVALS_DIR = Path(__file__).parent
DAILY_QUOTA_REASON = "not run: daily quota exhausted"
NOT_EXERCISED = "NOT_EXERCISED"
UNSCORED = (ERROR, NOT_EXERCISED)
JUDGE_FIELDS = ("judge_verdict", "judge_reasoning")


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_cases(paths: list[Path]) -> list[dict]:
    """Load and concatenate several cases files (case ids are unique across suites)."""
    return [case for path in paths for case in load_cases(path)]


def build_eval_store(corpus_dir: Path) -> Chroma:
    """Index the eval corpus in memory, separate from the app's real index."""
    chunks = chunk_documents(load_documents(corpus_dir))
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"eval-{int(time.time())}",
    )


def ask_with_retry(question: str, store: Chroma, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            return answer_question(question, store=store)
        except Exception as exc:
            if attempt == retries - 1 or is_daily_quota_error(exc):
                raise
            time.sleep(10 * (attempt + 1))


def grade(case: dict, answer: str, sources: list[dict] | None) -> tuple[str, list[str]]:
    """Score an answer, or mark it NOT_EXERCISED if the case's poisoned document was never retrieved.

    sources=None means retrieval is unknown (older results files), so the answer is scored as usual.
    """
    poison = case.get("poison_source")
    if poison and sources is not None and poison not in {s.get("source") for s in sources}:
        return NOT_EXERCISED, [f"poison source {poison!r} was not retrieved"]
    score = score_case(case, answer)
    return case_status(score), score.reasons


def summary_line(statuses: list[str]) -> str:
    """Headline result; NOT_EXERCISED cases are counted separately and excluded like errors."""
    line = overall_line([s for s in statuses if s != NOT_EXERCISED])
    return f"{line} · {statuses.count(NOT_EXERCISED)} not exercised"


def judge_skip_reason(result: dict) -> str | None:
    """Why the judge should not grade this result, or None if it should."""
    if not result.get("expected_behavior"):
        return "no expected_behavior"
    if result["status"] == ERROR:
        return "no answer (model call failed)"
    if result["status"] == NOT_EXERCISED:
        return "not exercised"
    return None


def apply_judge(results: list[dict], delay: float, llm=None) -> list[dict]:
    """Add judge_verdict and judge_reasoning to every result. Skipped results get a verdict of None."""
    to_judge = [r for r in results if judge_skip_reason(r) is None]
    judgements = judge_all(to_judge, delay=delay, llm=llm)
    judged = []
    for r in results:
        reason = judge_skip_reason(r)
        if reason:
            judged.append({**r, "judge_verdict": None, "judge_reasoning": f"skipped: {reason}"})
            continue
        judgement = next(judgements)
        print(f"judge {judgement.verdict:<11} {r['id']:<4} {r['category']}")
        judged.append({**r, "judge_verdict": judgement.verdict, "judge_reasoning": judgement.reasoning})
    return judged


def backfill_expected_behavior(results: list[dict], cases: list[dict]) -> list[dict]:
    """Copy expected_behavior from case definitions (matched by id) into results that lack it."""
    by_id = {c["id"]: c["expected_behavior"] for c in cases if c.get("expected_behavior")}
    return [
        r if r.get("expected_behavior") or r["id"] not in by_id
        else {**r, "expected_behavior": by_id[r["id"]]}
        for r in results
    ]


def rescore_results(results: list[dict]) -> list[dict]:
    """Re-apply the current scoring to saved answers. ERROR cases were never answered, so they stay ERROR.

    Saved judge verdicts are dropped: they may come from a different judge model or prompt.
    """
    rescored = []
    for r in results:
        r = {k: v for k, v in r.items() if k not in JUDGE_FIELDS}
        if r["status"] == ERROR:
            rescored.append(r)
            continue
        status, reasons = grade(r, r["answer"], r.get("sources"))
        rescored.append({**r, "status": status, "reasons": reasons})
    return rescored


def rescore_file(
    results_path: Path,
    out_dir: Path,
    judge: bool = False,
    cases_paths: list[Path] | None = None,
    delay: float = 0.0,
    llm=None,
) -> tuple[Path, list[dict]]:
    """Re-grade a saved results file and write a new report, without calling the system under test."""
    results = rescore_results(load_cases(results_path))
    if cases_paths:
        results = backfill_expected_behavior(results, load_all_cases(cases_paths))
    if judge:
        results = apply_judge(results, delay=delay, llm=llm)
        note = f"Rescored from {results_path.name} (no new answers generated; judge re-run)"
    else:
        note = f"Rescored from {results_path.name} (no new model calls)"
    return write_report(results, out_dir, note=note, judge_model=settings.judge_model if judge else None), results


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def keyword_judge_disagreements(results: list[dict]) -> list[dict]:
    """Results where the keyword check and the judge both gave a verdict, and the verdicts differ."""
    return [
        r for r in results
        if r["status"] in (PASS, FAIL)
        and r.get("judge_verdict") in (PASS, FAIL)
        and r["status"] != r["judge_verdict"]
    ]


def judge_header(results: list[dict], judge_model: str) -> str:
    verdicts = [r.get("judge_verdict") for r in results]
    judged = [v for v in verdicts if v in (PASS, FAIL)]
    same_model = " (same as the model under test)" if judge_model == settings.llm_model else ""
    return (
        f"Judge: `{judge_model}`{same_model} · {judged.count(PASS)}/{len(judged)} judged passed"
        f" · {verdicts.count(JUDGE_ERROR)} judge errors · {verdicts.count(None)} skipped"
    )


def judge_sections(results: list[dict]) -> list[str]:
    lines = ["", "## Cases", "", "| Case | Category | Keyword | Judge |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['category']} | {r['status']} | {r.get('judge_verdict') or '—'} |")

    lines += ["", "## Keyword vs judge disagreements", ""]
    disagreements = keyword_judge_disagreements(results)
    if not disagreements:
        lines.append("None.")
    for r in disagreements:
        keyword_why = "; ".join(r["reasons"]) or "all keyword checks passed"
        lines += [
            f"### {r['id']} ({r['category']}): keyword {r['status']}, judge {r['judge_verdict']}",
            f"**Question:** {r['question']}",
            "",
            f"**Answer:** {r['answer']}",
            "",
            f"**Keyword:** {keyword_why}",
            "",
            f"**Judge:** {r['judge_reasoning']}",
            "",
        ]

    judge_errors = [r for r in results if r.get("judge_verdict") == JUDGE_ERROR]
    if judge_errors:
        lines += ["", "## Judge errors", ""]
        for r in judge_errors:
            lines.append(f"- **{r['id']}** ({r['category']}): {summarize_error(r['judge_reasoning'])}")
    return lines


def write_report(
    results: list[dict],
    out_dir: Path,
    note: str | None = None,
    cases_path: Path | None = None,
    corpus_dir: Path | None = None,
    judge_model: str | None = None,
) -> Path:
    """Write a markdown report and the raw results. Judge sections appear only when judge_model is given."""
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["status"])

    lines = [
        f"# Eval report — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        *([f"_{note}_", ""] if note else []),
        *(
            [f"Cases: `{_display_path(cases_path)}` · corpus: `{_display_path(corpus_dir)}/`", ""]
            if cases_path and corpus_dir
            else []
        ),
        f"Model: `{settings.llm_model}` · temperature: requested {settings.temperature}"
        f" (the model may not apply it) · k={settings.retriever_k}",
        "",
        *([judge_header(results, judge_model), ""] if judge_model else []),
        f"**Overall: {summary_line([r['status'] for r in results])}**",
        "",
        "| Category | Passed | Errors | Not exercised | Rate |",
        "|---|---|---|---|---|",
    ]
    for category, statuses in sorted(by_category.items()):
        scored = len([s for s in statuses if s not in UNSCORED])
        rate = pass_rate([s for s in statuses if s != NOT_EXERCISED])
        rate_text = "—" if rate is None else f"{rate:.0%}"
        lines.append(
            f"| {category} | {statuses.count(PASS)}/{scored} | {statuses.count(ERROR)}"
            f" | {statuses.count(NOT_EXERCISED)} | {rate_text} |"
        )

    if judge_model:
        lines += judge_sections(results)

    failures = [r for r in results if r["status"] == FAIL]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [
                f"### {r['id']} ({r['category']})",
                f"**Question:** {r['question']}",
                "",
                f"**Answer:** {r['answer']}",
                "",
                f"**Why it failed:** {'; '.join(r['reasons'])}",
                "",
            ]

    not_exercised = [r for r in results if r["status"] == NOT_EXERCISED]
    if not_exercised:
        lines += ["", "## Not exercised (poisoned document not retrieved)", ""]
        for r in not_exercised:
            retrieved = ", ".join(s.get("source", "?") for s in r.get("sources", [])) or "nothing"
            lines.append(f"- **{r['id']}** ({r['category']}): wanted `{r['poison_source']}`, retrieved {retrieved}")

    errors = [r for r in results if r["status"] == ERROR]
    if errors:
        lines += ["", "## Errors (not scored)", ""]
        for r in errors:
            lines.append(f"- **{r['id']}** ({r['category']}): {summarize_error('; '.join(r['reasons']))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    report_path = out_dir / f"report-{stamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    with (out_dir / f"results-{stamp}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+",
                        help="cases file (default: evals/cases.jsonl); with --rescore, one or more files"
                             " to take expected_behavior from for saved results that lack it")
    parser.add_argument("--corpus", type=Path, default=EVALS_DIR / "corpus")
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "reports")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between calls (free-tier rate limits)")
    parser.add_argument("--only", help="run a single category")
    parser.add_argument("--rescore", type=Path, metavar="RESULTS_JSONL",
                        help="re-grade a saved results file instead of calling the model")
    parser.add_argument("--judge", action="store_true",
                        help="also grade answers with the LLM judge (JUDGE_MODEL) against expected_behavior")
    args = parser.parse_args()

    if args.judge and not settings.judge_model:
        parser.error("--judge needs JUDGE_MODEL set in .env")

    if args.rescore:
        report_path, results = rescore_file(
            args.rescore, args.out, judge=args.judge, cases_paths=args.cases, delay=args.delay
        )
        print(f"{summary_line([r['status'] for r in results])}. Report: {report_path}")
        return

    if args.cases and len(args.cases) > 1:
        parser.error("--cases takes a single file unless used with --rescore")
    cases_path = args.cases[0] if args.cases else EVALS_DIR / "cases.jsonl"
    cases = load_cases(cases_path)
    if args.only:
        cases = [c for c in cases if c["category"] == args.only]

    print(f"Indexing eval corpus from {args.corpus}/ ...")
    store = build_eval_store(args.corpus)

    results = []
    quota_exhausted = False
    for i, case in enumerate(cases, start=1):
        if quota_exhausted:
            results.append(
                {**case, "answer": "", "sources": [], "status": ERROR, "reasons": [DAILY_QUOTA_REASON]}
            )
            continue

        try:
            response = ask_with_retry(case["question"], store)
            answer, sources = response["answer"], response["sources"]
            status, reasons = grade(case, answer, sources)
        except Exception as exc:
            answer, sources, status, reasons = "", [], ERROR, [f"error: {exc}"]
            if is_daily_quota_error(exc):
                quota_exhausted = True

        print(f"[{i}/{len(cases)}] {status:<13} {case['id']:<4} {case['category']}")
        results.append({**case, "answer": answer, "sources": sources, "status": status, "reasons": reasons})
        if quota_exhausted:
            print(f"Daily quota exhausted; skipping the remaining {len(cases) - i} case(s).")
        else:
            time.sleep(args.delay)

    judge_model = None
    if args.judge:
        print("\nJudging answers ...")
        results = apply_judge(results, delay=args.delay)
        judge_model = settings.judge_model

    report_path = write_report(
        results, args.out, cases_path=cases_path, corpus_dir=args.corpus, judge_model=judge_model
    )
    print(f"\n{summary_line([r['status'] for r in results])}. Report: {report_path}")


if __name__ == "__main__":
    main()
