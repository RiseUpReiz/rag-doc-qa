"""Build a human-labelling template from saved eval results.

Usage (from the repo root):
    python -m evals.make_label_template evals/reports/results-<stamp>.jsonl [more results files...]
    python -m evals.make_label_template evals/reports/results-<stamp>.jsonl --cases evals/attacks/cases.jsonl

Writes one line per answer with human_label set to null. Fill each human_label in with
"PASS" or "FAIL" by hand, then run python -m evals.validate_judge. Judge output is
deliberately left out so it can't anchor the human labels.
"""
import argparse
import json
from pathlib import Path

from evals.run_eval import backfill_expected_behavior, load_all_cases, load_cases
from evals.scoring import ERROR

DEFAULT_OUT = Path(__file__).parent / "judge_validation" / "labels.jsonl"


def label_rows(results: list[dict], source: str) -> list[dict]:
    """One unlabelled row per answered result. ERROR results have no answer, so they are left out."""
    rows = []
    for r in results:
        if r.get("status") == ERROR:
            continue
        row = {"id": r["id"], "source": source, "question": r["question"]}
        if r.get("expected_behavior"):
            row["expected_behavior"] = r["expected_behavior"]
        row.update({"answer": r["answer"], "human_label": None, "note": ""})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="results-*.jsonl files from evals/reports")
    parser.add_argument("--cases", type=Path, nargs="+", default=[],
                        help="cases file(s) to fill in expected_behavior for results that lack it")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="overwrite an existing labels file")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        parser.error(f"{args.out} already exists and may contain labels; pass --force to overwrite it")

    cases = load_all_cases(args.cases)
    rows = []
    for path in args.results:
        results = backfill_expected_behavior(load_cases(path), cases)
        rows += label_rows(results, source=path.name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing = sum(1 for row in rows if "expected_behavior" not in row)
    print(f"Wrote {len(rows)} rows to {args.out}.")
    if missing:
        print(f"{missing} row(s) have no expected_behavior; the judge can't grade those (try --cases).")


if __name__ == "__main__":
    main()
