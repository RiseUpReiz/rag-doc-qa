"""Finish judging every run that carries a tag, resumably.

Usage (from the repo root):
    python -m evals.grade --tag final-ablation
    python -m evals.grade --tag final-ablation --dry-run

Finds the latest version of each tagged run (following rescored_from chains, as
evals.ablation does) and rescores each one that still has missing or JUDGE_ERROR judge
verdicts with --judge-missing-only, using the cases file recorded in the run. Prints the
number of judge calls needed before starting. If the judge's daily quota runs out it stops,
keeps what was judged, and prints what remains; running the same command later resumes.
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from evals.ablation import Run, load_runs
from evals.run_eval import EVALS_DIR, TAG_PATTERN, judge_calls_needed, rescore_file, stopped_by_daily_quota


@dataclass
class PendingRun:
    run: Run
    cases_path: Path
    calls: int


def plan(reports_dir: Path, tag: str) -> tuple[list[PendingRun], list[str]]:
    """Latest tagged runs that still need judge calls, plus problems that prevent grading a run."""
    pending, problems = [], []
    for run in load_runs(reports_dir, tag):
        cases_file = run.rows[0].get("cases_file")
        if not cases_file or not Path(cases_file).is_file():
            problems.append(f"{run.path.name}: cases file {cases_file!r} not found; skipped")
            continue
        calls = judge_calls_needed(run.path, [Path(cases_file)])
        if calls:
            pending.append(PendingRun(run, Path(cases_file), calls))
    return pending, problems


def remaining_calls(reports_dir: Path, tag: str) -> int:
    return sum(p.calls for p in plan(reports_dir, tag)[0])


def grade_tag(reports_dir: Path, tag: str, delay: float = 4.0, llm=None, dry_run: bool = False) -> int:
    """Judge what is missing across the tag's runs. Returns how many judge calls still remain."""
    pending, problems = plan(reports_dir, tag)
    for problem in problems:
        print(problem)
    total = sum(p.calls for p in pending)
    if not total:
        print(f"Nothing to grade: every run tagged {tag!r} already has its judge verdicts.")
        return 0

    print(f"{total} judge call(s) needed across {len(pending)} run(s):")
    for p in pending:
        print(f"  {p.run.path.name}: {p.calls}")
    if dry_run:
        return total

    for i, p in enumerate(pending, start=1):
        if i > 1 and delay:
            time.sleep(delay)
        print(f"\n[{i}/{len(pending)}] {p.run.path.name} ({p.calls} call(s))")
        _, results = rescore_file(
            p.run.path, reports_dir, cases_paths=[p.cases_path], delay=delay, llm=llm, judge_missing_only=True
        )
        if stopped_by_daily_quota(results):
            left = remaining_calls(reports_dir, tag)
            print(f"\nJudge daily quota exhausted. {left} judge call(s) remain;"
                  " run the same command again later to resume.")
            return left

    left = remaining_calls(reports_dir, tag)
    if left:
        print(f"\nDone, but {left} judge call(s) still failed (see JUDGE_ERROR rows); run again to retry them.")
    else:
        print(f"\nDone: every run tagged {tag!r} now has its judge verdicts.")
    return left


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reports", type=Path, default=EVALS_DIR / "reports")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between judge calls (free-tier rate limits)")
    parser.add_argument("--dry-run", action="store_true", help="only print how many judge calls are needed")
    args = parser.parse_args()

    if not TAG_PATTERN.fullmatch(args.tag):
        parser.error("--tag may only contain letters, digits, '.', '_' and '-'")
    if not settings.judge_model:
        parser.error("JUDGE_MODEL is not set in .env")
    if not load_runs(args.reports, args.tag):
        parser.error(f"no results in {args.reports} carry the tag {args.tag!r}")

    grade_tag(args.reports, args.tag, delay=args.delay, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
