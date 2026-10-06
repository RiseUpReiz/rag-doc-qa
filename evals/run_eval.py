"""Run the safety and quality eval suite against the RAG pipeline.

Usage (from the repo root):
    python -m evals.run_eval
    python -m evals.run_eval --delay 6 --only indirect_injection
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


def ask_with_retry(question: str, store: Chroma, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            return answer_question(question, store=store)["answer"]
        except Exception as exc:
            if attempt == retries - 1 or is_daily_quota_error(exc):
                raise
            time.sleep(10 * (attempt + 1))


def write_report(results: list[dict], out_dir: Path) -> Path:
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["status"])

    lines = [
        f"# Eval report — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Model: `{settings.llm_model}` · temperature: requested {settings.temperature}"
        f" (the model may not apply it) · k={settings.retriever_k}",
        "",
        f"**Overall: {overall_line([r['status'] for r in results])}**",
        "",
        "| Category | Passed | Errors | Rate |",
        "|---|---|---|---|",
    ]
    for category, statuses in sorted(by_category.items()):
        scored = len(statuses) - statuses.count(ERROR)
        rate = pass_rate(statuses)
        rate_text = "—" if rate is None else f"{rate:.0%}"
        lines.append(
            f"| {category} | {statuses.count(PASS)}/{scored} | {statuses.count(ERROR)} | {rate_text} |"
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
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c["category"] == args.only]

    print(f"Indexing eval corpus from {args.corpus}/ ...")
    store = build_eval_store(args.corpus)

    results = []
    quota_exhausted = False
    for i, case in enumerate(cases, start=1):
        if quota_exhausted:
            results.append({**case, "answer": "", "status": ERROR, "reasons": [DAILY_QUOTA_REASON]})
            continue

        try:
            answer = ask_with_retry(case["question"], store)
            score = score_case(case, answer)
            reasons = score.reasons
        except Exception as exc:
            answer, score, reasons = "", None, [f"error: {exc}"]
            if is_daily_quota_error(exc):
                quota_exhausted = True

        status = case_status(score)
        print(f"[{i}/{len(cases)}] {status:<5} {case['id']:<4} {case['category']}")
        results.append({**case, "answer": answer, "status": status, "reasons": reasons})
        if quota_exhausted:
            print(f"Daily quota exhausted; skipping the remaining {len(cases) - i} case(s).")
        else:
            time.sleep(args.delay)

    report_path = write_report(results, args.out)
    print(f"\n{overall_line([r['status'] for r in results])}. Report: {report_path}")


if __name__ == "__main__":
    main()
