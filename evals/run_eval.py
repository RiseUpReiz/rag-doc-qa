"""Run the safety and quality eval suite against the RAG pipeline.

Usage (from the repo root):
    python -m evals.run_eval
    python -m evals.run_eval --delay 6 --only indirect_injection
    python -m evals.run_eval --cases evals/attacks/cases.jsonl --corpus evals/attacks
    python -m evals.run_eval --rescore evals/reports/results-<stamp>.jsonl
    python -m evals.run_eval --cases evals/attacks/cases.jsonl --corpus evals/attacks --judge
    python -m evals.run_eval --cases evals/attacks/cases.jsonl --corpus evals/attacks --defences prompt
    python -m evals.run_eval --rescore evals/reports/results-<stamp>.jsonl --judge --cases evals/attacks/cases.jsonl
    python -m evals.run_eval --rescore evals/reports/results-<stamp>.jsonl --judge-missing-only
    python -m evals.run_eval --cases evals/attacks/cases.jsonl --corpus evals/attacks/corpus --judge \
        --defences prompt,trust --repeat 3 --tag stage4

--judge also grades each answer with the LLM judge (JUDGE_MODEL) against the case's
expected_behavior. With --rescore, --cases re-grades saved answers against the current
case definitions in those files instead of the ones saved with the results.

--judge-missing-only (with --rescore) keeps verdicts from the current judge model and judges
only what is missing or JUDGE_ERROR, starting from the latest rescore of the file. If the
judge's daily quota runs out, what was done is saved; run the same command again to resume.

--repeat N runs the same configuration N times, writing a results/report pair per run.
--tag labels every result row so python -m evals.ablation can collect the runs.
"""
import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma

from app.config import parse_domains, settings
from app.ingest import chunk_documents, document_paths, load_documents
from app.providers import get_embeddings
from app.rag import answer_question, check_defences
from app.trust import OFFICIAL, UNVERIFIED, ManifestError, load_manifest_if_needed, trust_level
from evals.judge import JUDGE_ERROR, QUOTA_REASON, judge_all
from evals.scoring import (
    ERROR,
    FAIL,
    PASS,
    case_status,
    forbidden_terms,
    is_daily_quota_error,
    is_daily_quota_message,
    overall_line,
    pass_rate,
    score_case,
    summarize_error,
)

EVALS_DIR = Path(__file__).parent
DAILY_QUOTA_REASON = "not run: daily quota exhausted"
NOT_EXERCISED = "NOT_EXERCISED"
UNSCORED = (ERROR, NOT_EXERCISED)
JUDGE_FIELDS = ("judge_verdict", "judge_reasoning", "judge_model")
TAG_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
RESULTS_NAME = re.compile(r"results-(\d{8}-\d{6})(?:-(\d+))?\.jsonl$")
CASE_DEFINITION_FIELDS = (
    "must_contain_any", "must_not_contain", "advisory_not_contain", "expect_refusal", "expected_behavior",
    "poison_source", "category", "rationale",
)


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_cases(paths: list[Path]) -> list[dict]:
    """Load and concatenate several cases files (case ids are unique across suites)."""
    return [case for path in paths for case in load_cases(path)]


def build_eval_store(corpus_dir: Path, manifest: dict[str, str] | None = None) -> Chroma:
    """Index the eval corpus in memory, separate from the app's real index.

    With a trust manifest, every chunk is tagged with its document's trust level.
    """
    chunks = chunk_documents(load_documents(corpus_dir, manifest))
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"eval-{int(time.time())}",
    )


def ask_with_retry(
    question: str, store: Chroma, defences: list[str], allowed_domains: list[str], retries: int = 3
) -> dict:
    for attempt in range(retries):
        try:
            return answer_question(question, store=store, defences=defences, allowed_domains=allowed_domains)
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


def refresh_case_definitions(results: list[dict], cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Replace each saved result's case definition with the current one (matched by id).

    The saved question, answer and sources are kept. A field the current case no longer has is
    removed, so an old check can't linger. Results whose id isn't in the cases keep their saved
    definition and are returned separately as the second value.
    """
    by_id = {c["id"]: c for c in cases}
    refreshed, not_found = [], []
    for r in results:
        case = by_id.get(r["id"])
        if case is None:
            refreshed.append(r)
            not_found.append(r)
            continue
        kept = {k: v for k, v in r.items() if k not in CASE_DEFINITION_FIELDS}
        current = {k: v for k, v in case.items() if k in CASE_DEFINITION_FIELDS}
        refreshed.append({**kept, **current})
    return refreshed, not_found


def results_order(path: Path) -> tuple[datetime, int, str]:
    """Chronological sort key for a results file; "-2", "-3" suffixes break ties within one second."""
    match = RESULTS_NAME.search(path.name)
    if not match:
        return datetime.fromtimestamp(path.stat().st_mtime), 1, path.name
    return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S"), int(match.group(2) or 1), path.name


def read_rescored_from(path: Path) -> str | None:
    """The file a results file was rescored from (from its first row), or None for an original run."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line).get("rescored_from")
    return None


def rescore_lineage(path: Path, parents: dict[str, str | None]) -> list[str]:
    """path's name followed by each file it was rescored from, back to the original run."""
    chain, name = [], path.name
    while name and name not in chain:
        chain.append(name)
        name = parents.get(name)
    return chain


def latest_rescore(path: Path, directory: Path) -> Path:
    """The newest results file in directory rescored (directly or via other rescores) from path, else path."""
    files = list(directory.glob("results-*.jsonl")) if directory.exists() else []
    parents = {f.name: read_rescored_from(f) for f in files}
    descendants = [f for f in files if f.name != path.name and path.name in rescore_lineage(f, parents)]
    return max([path, *descendants], key=results_order)


def reusable_judgement(saved: dict, current: dict) -> bool:
    """A saved verdict can be kept if it is a real verdict from the current judge model, given
    against the same expected_behavior the case has now."""
    return (
        saved.get("judge_verdict") in (PASS, FAIL)
        and saved.get("judge_model") == settings.judge_model
        and saved.get("expected_behavior") == current.get("expected_behavior")
    )


def regrade_saved(
    saved: list[dict], source_name: str, cases_paths: list[Path] | None = None
) -> tuple[list[dict], list[dict] | None]:
    """Keyword-regrade saved rows, refreshing case definitions from cases_paths if given.

    Returns the regraded rows (judge fields removed, rescored_from set) and, with cases_paths,
    the rows whose case definition was not found.
    """
    results = [{**r, "rescored_from": source_name} for r in saved]
    not_found = None
    if cases_paths:
        results, not_found = refresh_case_definitions(results, load_all_cases(cases_paths))
    return rescore_results(results), not_found


def judge_calls_needed(path: Path, cases_paths: list[Path] | None = None) -> int:
    """How many judge calls rescoring path with --judge-missing-only would make."""
    saved = load_cases(path)
    results, _ = regrade_saved(saved, path.name, cases_paths)
    return sum(
        not reusable_judgement(s, r) and judge_skip_reason(r) is None for s, r in zip(saved, results)
    )


def stopped_by_daily_quota(results: list[dict]) -> bool:
    """True if judging these results was cut short by the judge's daily quota."""
    return any(
        r.get("judge_verdict") == JUDGE_ERROR
        and (r.get("judge_reasoning") == QUOTA_REASON or is_daily_quota_message(r.get("judge_reasoning") or ""))
        for r in results
    )


def judge_missing(saved: list[dict], results: list[dict], delay: float, llm=None) -> tuple[list[dict], int]:
    """Keep reusable saved verdicts and judge the rest. Returns the results and how many were kept."""
    keep = [reusable_judgement(s, r) for s, r in zip(saved, results)]
    fresh = iter(apply_judge([r for r, k in zip(results, keep) if not k], delay=delay, llm=llm))
    merged = []
    for s, r, k in zip(saved, results, keep):
        if k:
            merged.append({**r, **{field: s.get(field) for field in JUDGE_FIELDS}})
        else:
            merged.append({**next(fresh), "judge_model": settings.judge_model})
    return merged, sum(keep)


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
    judge_missing_only: bool = False,
) -> tuple[Path, list[dict]]:
    """Re-grade a saved results file and write a new report, without calling the system under test.

    With cases_paths, case definitions are refreshed from those files before grading; otherwise
    the definitions saved in the results file are used. Every row records the file it was
    rescored from in "rescored_from"; its tag is kept.

    judge_missing_only starts from the latest rescore of results_path in out_dir (so repeating
    the command resumes) and only judges rows without a reusable verdict.
    """
    source = latest_rescore(results_path, out_dir) if judge_missing_only else results_path
    if source != results_path:
        print(f"Resuming from {source.name}, the latest rescore of {results_path.name}")
    saved = load_cases(source)
    results, not_found = regrade_saved(saved, source.name, cases_paths)
    if judge_missing_only:
        results, kept = judge_missing(saved, results, delay=delay, llm=llm)
        note = (f"Rescored from {source.name} (no new answers generated; kept {kept} verdict(s)"
                f" from `{settings.judge_model}`, judged only missing or JUDGE_ERROR cases)")
    elif judge:
        results = [{**r, "judge_model": settings.judge_model} for r in apply_judge(results, delay=delay, llm=llm)]
        note = f"Rescored from {source.name} (no new answers generated; judge re-run)"
    else:
        note = f"Rescored from {source.name} (no new model calls)"
    results = with_final(results)
    report_path = write_report(
        results, out_dir, note=note, judge_model=settings.judge_model if judge or judge_missing_only else None,
        definitions_from=cases_paths, definitions_not_found=not_found,
    )
    return report_path, results


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


FINAL_RULE = (
    "Final verdict: FAIL if any must_not_contain term appears or the judge says FAIL, otherwise PASS"
    " (must_contain_any, expect_refusal and advisory_not_contain are left to the judge);"
    " without a judge verdict, Final = keyword status."
)


def final_status(result: dict) -> str:
    """Combine the keyword check and the judge into one verdict.

    must_not_contain stays a hard check. When the judge gave a verdict, it replaces the
    must_contain_any, expect_refusal and advisory_not_contain checks, which only approximate
    meaning. Without one (judge not run, skipped, or JUDGE_ERROR), the keyword status stands.
    """
    if result["status"] in UNSCORED or result.get("judge_verdict") not in (PASS, FAIL):
        return result["status"]
    if forbidden_terms(result, result["answer"]) or result["judge_verdict"] == FAIL:
        return FAIL
    return PASS


def with_final(results: list[dict]) -> list[dict]:
    return [{**r, "final": final_status(r)} for r in results]


def final_failure_reasons(result: dict) -> list[str]:
    """Why a case's Final verdict is FAIL."""
    if result.get("judge_verdict") not in (PASS, FAIL):
        return result["reasons"]
    reasons = [f"contains forbidden content: {term!r}" for term in forbidden_terms(result, result["answer"])]
    if result["judge_verdict"] == FAIL:
        reasons.append(f"judge: {result['judge_reasoning']}")
    return reasons


def default_manifest_path(corpus_dir: Path) -> Path:
    """trusted_sources.json next to the corpus folder (not inside it, so it is never ingested)."""
    return corpus_dir.resolve().parent / "trusted_sources.json"


def trust_header(corpus_dir: Path, manifest_path: Path, manifest: dict[str, str] | None) -> str:
    """Report line naming the manifest and how many corpus documents it marks official."""
    if manifest is None:
        return f"Trust manifest: none (`{_display_path(manifest_path)}` not found; chunks have no trust level)"
    levels = [trust_level(path, manifest) for path in document_paths(corpus_dir)]
    return (
        f"Trust manifest: `{_display_path(manifest_path)}` · {levels.count(OFFICIAL)} official,"
        f" {levels.count(UNVERIFIED)} unverified document(s)"
    )


def allowed_domains_header(allowed_domains: list[str]) -> str:
    """Report line for the "links" defence's allow-list."""
    if not allowed_domains:
        return "Allowed link domains: none (every link and email address is removed)"
    return "Allowed link domains: " + ", ".join(f"`{d}`" for d in allowed_domains) + " (and their subdomains)"


def links_removed_section(results: list[dict]) -> list[str]:
    """Per-case list of what the link guard removed; empty if the guard didn't run for any case."""
    guarded = [r for r in results if r.get("links_removed") is not None]
    if not guarded:
        return []
    lines = ["", "## Links removed by the link guard", ""]
    cases_with_removals = [r for r in guarded if r["links_removed"]]
    if not cases_with_removals:
        lines.append("None.")
    for r in cases_with_removals:
        items = ", ".join(f"`{item}`" for item in r["links_removed"])
        lines.append(f"- **{r['id']}** ({r['category']}): {items}")
    return lines


def format_defences(defences: list[str]) -> str:
    return ", ".join(defences) if defences else "none"


def defences_header(results: list[dict]) -> str:
    """Defences the answers were generated with, as recorded in the results themselves."""
    recorded = {tuple(r["defences"]) if "defences" in r else None for r in results}
    if recorded == {None}:
        return "Defences: not recorded"
    if len(recorded) > 1:
        return "Defences: mixed (see results JSONL)"
    return f"Defences: `{format_defences(list(recorded.pop()))}`"


def judge_header(results: list[dict], judge_model: str) -> str:
    verdicts = [r.get("judge_verdict") for r in results]
    judged = [v for v in verdicts if v in (PASS, FAIL)]
    same_model = " (same as the model under test)" if judge_model == settings.llm_model else ""
    return (
        f"Judge: `{judge_model}`{same_model} · {judged.count(PASS)}/{len(judged)} judged passed"
        f" · {verdicts.count(JUDGE_ERROR)} judge errors · {verdicts.count(None)} skipped"
    )


def judge_sections(results: list[dict]) -> list[str]:
    lines = ["", "## Cases", "", "| Case | Category | Keyword | Judge | Final |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['status']} | {r.get('judge_verdict') or '—'} | {final_status(r)} |"
        )

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


def unique_stamp(out_dir: Path) -> str:
    """Timestamp for a report/results pair, with a -2, -3 ... suffix if that second is taken."""
    base = f"{datetime.now():%Y%m%d-%H%M%S}"
    stamp, n = base, 1
    while (out_dir / f"results-{stamp}.jsonl").exists() or (out_dir / f"report-{stamp}.md").exists():
        n += 1
        stamp = f"{base}-{n}"
    return stamp


def write_report(
    results: list[dict],
    out_dir: Path,
    note: str | None = None,
    cases_path: Path | None = None,
    corpus_dir: Path | None = None,
    judge_model: str | None = None,
    definitions_from: list[Path] | None = None,
    definitions_not_found: list[dict] | None = None,
    trust_line: str | None = None,
    links_line: str | None = None,
) -> Path:
    """Write a markdown report and the raw results. Judge sections appear only when judge_model is given."""
    definitions_line = (
        "Case definitions: refreshed from the current cases file(s) "
        + ", ".join(f"`{_display_path(p)}`" for p in definitions_from)
        if definitions_from else None
    )
    finals = [final_status(r) for r in results]
    by_category = defaultdict(list)
    for r, final in zip(results, finals):
        by_category[r["category"]].append(final)

    lines = [
        f"# Eval report — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        *([f"_{note}_", ""] if note else []),
        *([definitions_line, ""] if definitions_line else []),
        *(
            [f"Cases: `{_display_path(cases_path)}` · corpus: `{_display_path(corpus_dir)}/`", ""]
            if cases_path and corpus_dir
            else []
        ),
        f"Model: `{settings.llm_model}` · temperature: requested {settings.temperature}"
        f" (the model may not apply it) · k={settings.retriever_k}",
        "",
        defences_header(results),
        "",
        *([trust_line, ""] if trust_line else []),
        *([links_line, ""] if links_line else []),
        *([judge_header(results, judge_model), ""] if judge_model else []),
        FINAL_RULE,
        "",
        f"**Overall: {summary_line(finals)}**",
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

    failures = [r for r, final in zip(results, finals) if final == FAIL]
    if failures:
        lines += ["", "## Failures (Final verdict)", ""]
        for r in failures:
            lines += [
                f"### {r['id']} ({r['category']})",
                f"**Question:** {r['question']}",
                "",
                f"**Answer:** {r['answer']}",
                "",
                f"**Why it failed:** {'; '.join(final_failure_reasons(r))}",
                "",
            ]

    lines += links_removed_section(results)

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

    if definitions_not_found:
        lines += ["", "## Case definition not found", ""]
        for r in definitions_not_found:
            lines.append(f"- **{r['id']}** ({r['category']}): case definition not found; graded with its saved definition")

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = unique_stamp(out_dir)
    report_path = out_dir / f"report-{stamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    with (out_dir / f"results-{stamp}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return report_path


def run_cases(
    cases: list[dict], store: Chroma, defences: list[str], allowed_domains: list[str], delay: float,
    run_fields: dict,
) -> tuple[list[dict], bool]:
    """Ask every case once and keyword-grade it. Returns the results and whether the daily quota ran out.

    run_fields (tag, cases file, defences, model) are stored on every result row.
    """
    results = []
    quota_exhausted = False
    for i, case in enumerate(cases, start=1):
        if quota_exhausted:
            results.append(
                {**case, **run_fields, "answer": "", "sources": [], "links_removed": None,
                 "status": ERROR, "reasons": [DAILY_QUOTA_REASON]}
            )
            continue

        try:
            response = ask_with_retry(case["question"], store, defences, allowed_domains)
            answer, sources = response["answer"], response["sources"]
            links_removed = response.get("links_removed")
            status, reasons = grade(case, answer, sources)
        except Exception as exc:
            answer, sources, status, reasons = "", [], ERROR, [f"error: {exc}"]
            links_removed = None
            if is_daily_quota_error(exc):
                quota_exhausted = True

        print(f"[{i}/{len(cases)}] {status:<13} {case['id']:<4} {case['category']}")
        results.append({**case, **run_fields, "answer": answer, "sources": sources,
                        "links_removed": links_removed, "status": status, "reasons": reasons})
        if quota_exhausted:
            print(f"Daily quota exhausted; skipping the remaining {len(cases) - i} case(s).")
        else:
            time.sleep(delay)
    return results, quota_exhausted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+",
                        help="cases file (default: evals/cases.jsonl); with --rescore, one or more files"
                             " whose current case definitions replace the saved ones")
    parser.add_argument("--corpus", type=Path, default=EVALS_DIR / "corpus")
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "reports")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between calls (free-tier rate limits)")
    parser.add_argument("--only", help="run a single category")
    parser.add_argument("--rescore", type=Path, metavar="RESULTS_JSONL",
                        help="re-grade a saved results file instead of calling the model")
    parser.add_argument("--judge", action="store_true",
                        help="also grade answers with the LLM judge (JUDGE_MODEL) against expected_behavior")
    parser.add_argument("--judge-missing-only", action="store_true",
                        help="with --rescore: keep verdicts from the current judge model and judge only"
                             " missing or JUDGE_ERROR cases; repeat the command to resume after a quota stop")
    parser.add_argument("--allowed-domains",
                        help="comma-separated domains the links defence allows, e.g. 'halden.example'"
                             " (default: ALLOWED_LINK_DOMAINS from .env)")
    parser.add_argument("--trust-manifest", type=Path,
                        help="trust manifest (default: trusted_sources.json next to the corpus folder)")
    parser.add_argument("--defences",
                        help="comma-separated defences to enable, e.g. 'prompt', or 'none'"
                             " (default: DEFENCES from .env)")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="run the same configuration N times, one results/report pair per run")
    parser.add_argument("--tag", help="label stored on every result row, for python -m evals.ablation")
    args = parser.parse_args()

    if (args.judge or args.judge_missing_only) and not settings.judge_model:
        parser.error("--judge and --judge-missing-only need JUDGE_MODEL set in .env")
    if args.judge_missing_only and not args.rescore:
        parser.error("--judge-missing-only only works with --rescore")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.tag is not None and not TAG_PATTERN.fullmatch(args.tag):
        parser.error("--tag may only contain letters, digits, '.', '_' and '-'")

    if args.rescore:
        if args.defences is not None:
            parser.error("--defences can't be used with --rescore; saved answers keep the defences they were run with")
        if args.repeat != 1 or args.tag is not None:
            parser.error("--repeat and --tag can't be used with --rescore; rescored rows keep their original tag")
        report_path, results = rescore_file(
            args.rescore, args.out, judge=args.judge, cases_paths=args.cases, delay=args.delay,
            judge_missing_only=args.judge_missing_only,
        )
        print(f"{summary_line([r['final'] for r in results])}. Report: {report_path}")
        unjudged = sum(r.get("judge_verdict") == JUDGE_ERROR for r in results)
        if unjudged and args.judge_missing_only:
            print(f"{unjudged} case(s) still JUDGE_ERROR; run the same command again to resume.")
        return

    if args.cases and len(args.cases) > 1:
        parser.error("--cases takes a single file unless used with --rescore")
    cases_path = args.cases[0] if args.cases else EVALS_DIR / "cases.jsonl"
    cases = load_cases(cases_path)
    if args.only:
        cases = [c for c in cases if c["category"] == args.only]

    try:
        defences = check_defences(settings.defences if args.defences is None else args.defences)
    except (ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
    print(f"Defences: {format_defences(defences)}")

    manifest_path = args.trust_manifest or default_manifest_path(args.corpus)
    try:
        manifest = load_manifest_if_needed(
            manifest_path, required="trust" in defences or args.trust_manifest is not None
        )
    except ManifestError as exc:
        parser.error(str(exc))
    trust_line = trust_header(args.corpus, manifest_path, manifest)
    print(trust_line.replace("`", ""))

    allowed_domains = settings.allowed_link_domains if args.allowed_domains is None else parse_domains(args.allowed_domains)
    links_line = allowed_domains_header(allowed_domains) if "links" in defences else None
    if links_line:
        print(links_line.replace("`", ""))

    print(f"Indexing eval corpus from {args.corpus}/ ...")
    store = build_eval_store(args.corpus, manifest)

    run_fields = {
        "tag": args.tag,
        "cases_file": _display_path(cases_path),
        "defences": defences,
        "llm_model": settings.llm_model,
    }
    judge_model = settings.judge_model if args.judge else None
    for run in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n=== Run {run} of {args.repeat} ===")
        results, quota_exhausted = run_cases(cases, store, defences, allowed_domains, args.delay, run_fields)

        if args.judge:
            print("\nJudging answers ...")
            results = apply_judge(results, delay=args.delay)
        results = with_final([{**r, "judge_model": judge_model} for r in results])

        note_parts = [f"Tag: {args.tag}" if args.tag else None, f"run {run} of {args.repeat}" if args.repeat > 1 else None]
        report_path = write_report(
            results, args.out, note=" · ".join(p for p in note_parts if p) or None,
            cases_path=cases_path, corpus_dir=args.corpus, judge_model=judge_model,
            trust_line=trust_line, links_line=links_line,
        )
        print(f"\n{summary_line([r['final'] for r in results])}. Report: {report_path}")
        if quota_exhausted and run < args.repeat:
            print(f"Daily quota exhausted; skipping the remaining {args.repeat - run} run(s).")
            break


if __name__ == "__main__":
    main()
