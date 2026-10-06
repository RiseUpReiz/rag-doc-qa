"""Run the safety and quality eval suite against the RAG pipeline.

Usage (from the repo root):
    python -m evals.run_eval
    python -m evals.run_eval --delay 6 --only indirect_injection
    python -m evals.run_eval --cases evals/attacks/cases.jsonl --corpus evals/attacks
    python -m evals.run_eval --rescore evals/reports/results-<stamp>.jsonl
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


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def rescore_results(results: list[dict]) -> list[dict]:
    """Re-apply the current scoring to saved answers. ERROR cases were never answered, so they stay ERROR."""
    rescored = []
    for r in results:
        if r["status"] == ERROR:
            rescored.append(r)
            continue
        status, reasons = grade(r, r["answer"], r.get("sources"))
        rescored.append({**r, "status": status, "reasons": reasons})
    return rescored


def rescore_file(results_path: Path, out_dir: Path) -> tuple[Path, list[dict]]:
    """Re-grade a saved results file and write a new report, without calling the model."""
    results = rescore_results(load_cases(results_path))
    note = f"Rescored from {results_path.name} (no new model calls)"
    return write_report(results, out_dir, note=note), results


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def write_report(
    results: list[dict],
    out_dir: Path,
    note: str | None = None,
    cases_path: Path | None = None,
    corpus_dir: Path | None = None,
) -> Path:
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
    parser.add_argument("--cases", type=Path, default=EVALS_DIR / "cases.jsonl")
    parser.add_argument("--corpus", type=Path, default=EVALS_DIR / "corpus")
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "reports")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between calls (free-tier rate limits)")
    parser.add_argument("--only", help="run a single category")
    parser.add_argument("--rescore", type=Path, metavar="RESULTS_JSONL",
                        help="re-grade a saved results file instead of calling the model")
    args = parser.parse_args()

    if args.rescore:
        report_path, results = rescore_file(args.rescore, args.out)
        print(f"{summary_line([r['status'] for r in results])}. Report: {report_path}")
        return

    cases = load_cases(args.cases)
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

    report_path = write_report(results, args.out, cases_path=args.cases, corpus_dir=args.corpus)
    print(f"\n{summary_line([r['status'] for r in results])}. Report: {report_path}")


if __name__ == "__main__":
    main()
