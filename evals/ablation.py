"""Summarise repeated eval runs across defence configurations.

Usage (from the repo root):
    python -m evals.ablation --tag stage4

Collects every results file in evals/reports/ whose rows carry the tag (see run_eval --tag),
groups the runs by cases file and defences, and writes evals/reports/ablation-<tag>.md.
Passes are counted on the Final verdict. A rescored file records the file it came from
("rescored_from"); any file a later file was rescored from is ignored, so each run is counted
once with its latest grading. Runs that still have JUDGE_ERROR rows are flagged as incomplete
and left out of the min-max.
"""
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import VALID_DEFENCES
from evals.judge import JUDGE_ERROR
from evals.run_eval import EVALS_DIR, NOT_EXERCISED, TAG_PATTERN, rescore_lineage, results_order
from evals.scoring import ERROR, PASS

CONFIG_ORDER = [(), ("prompt",), ("prompt", "trust"), ("prompt", "trust", "links")]


@dataclass
class Run:
    path: Path
    rows: list[dict]

    @property
    def order(self) -> tuple[datetime, int, str]:
        return results_order(self.path)

    @property
    def started(self) -> datetime:
        return self.order[0]

    @property
    def rescored_from(self) -> str | None:
        return self.rows[0].get("rescored_from")

    @property
    def judge_errors(self) -> int:
        return sum(r.get("judge_verdict") == JUDGE_ERROR for r in self.rows)

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        keys = {(r.get("cases_file"), config_key(r.get("defences") or [])) for r in self.rows}
        if len(keys) != 1:
            raise ValueError(f"{self.path.name} mixes cases files or defences: {sorted(keys)}")
        return keys.pop()


def config_key(defences: list[str]) -> tuple[str, ...]:
    """Defences in canonical order, so 'trust,prompt' and 'prompt,trust' group together."""
    return tuple(d for d in VALID_DEFENCES if d in defences)


def config_label(config: tuple[str, ...]) -> str:
    return "+".join(config) or "none"


def final(row: dict) -> str:
    return row.get("final", row["status"])


def load_runs(reports_dir: Path, tag: str) -> list[Run]:
    """Tagged runs in reports_dir, oldest first, each counted once with its latest grading.

    A file that a later file was rescored from is dropped. If one run was rescored more than
    once separately, only the newest rescore is kept.
    """
    tagged = []
    for path in sorted(reports_dir.glob("results-*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows = [r for r in (json.loads(line) for line in f if line.strip()) if r.get("tag") == tag]
        if rows:
            tagged.append(Run(path, rows))

    superseded = {run.rescored_from for run in tagged}
    parents = {run.path.name: run.rescored_from for run in tagged}
    newest: dict[str, Run] = {}
    for run in tagged:
        if run.path.name in superseded:
            continue
        original = rescore_lineage(run.path, parents)[-1]
        if original not in newest or run.order > newest[original].order:
            newest[original] = run
    return sorted(newest.values(), key=lambda run: run.order)


def group_runs(runs: list[Run]) -> dict[str, dict[tuple[str, ...], list[Run]]]:
    """{cases file: {defence config: [runs]}}, configs in the standard ablation order."""
    grouped: dict[str, dict[tuple[str, ...], list[Run]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        cases_file, config = run.key
        grouped[cases_file][config].append(run)
    return {
        suite: dict(sorted(configs.items(), key=lambda item: _config_rank(item[0])))
        for suite, configs in sorted(grouped.items())
    }


def _config_rank(config: tuple[str, ...]) -> tuple[int, str]:
    if config in CONFIG_ORDER:
        return CONFIG_ORDER.index(config), ""
    return len(CONFIG_ORDER), config_label(config)


def _incomplete(runs: list[Run]) -> tuple[int, int]:
    rows = [r for run in runs for r in run.rows]
    return sum(final(r) == ERROR for r in rows), sum(final(r) == NOT_EXERCISED for r in rows)


def suite_table(configs: dict[tuple[str, ...], list[Run]]) -> list[str]:
    """One row per defence configuration. Runs with JUDGE_ERROR rows are marked and left out of min-max."""
    sizes = {len(run.rows) for runs in configs.values() for run in runs}
    lines = [
        "| Defences | Runs | Final passed per run | Min–max | Errors | Not exercised | Judge errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for config, runs in configs.items():
        passed = [sum(final(r) == PASS for r in run.rows) for run in runs]
        per_run = ", ".join(
            (str(p) if len(sizes) == 1 else f"{p}/{len(run.rows)}") + (" (incomplete)" if run.judge_errors else "")
            for p, run in zip(passed, runs)
        )
        complete = [p for p, run in zip(passed, runs) if not run.judge_errors]
        range_text = f"{min(complete)}–{max(complete)}" if complete else "—"
        errors, not_exercised = _incomplete(runs)
        judge_errors = sum(run.judge_errors for run in runs)
        flag = " ⚠" if errors or not_exercised or judge_errors else ""
        lines.append(
            f"| {config_label(config)}{flag} | {len(runs)} | {per_run} | {range_text}"
            f" | {errors} | {not_exercised} | {judge_errors} |"
        )
    return lines


def incomplete_notes(configs: dict[tuple[str, ...], list[Run]]) -> list[str]:
    notes = []
    for config, runs in configs.items():
        errors, not_exercised = _incomplete(runs)
        if errors or not_exercised:
            notes.append(
                f"- ⚠ **{config_label(config)}**: {errors} ERROR and {not_exercised} NOT_EXERCISED result(s)"
                f" across {len(runs)} run(s). Those cases count as not passed, so this row's numbers are incomplete."
            )
        unjudged = [run for run in runs if run.judge_errors]
        if unjudged:
            names = ", ".join(f"`{run.path.name}` ({run.judge_errors})" for run in unjudged)
            notes.append(
                f"- ⚠ **{config_label(config)}**: {len(unjudged)} of {len(runs)} run(s) still have JUDGE_ERROR rows:"
                f" {names}. Their Final counts are shown but left out of min–max. Finish them with"
                " `python -m evals.run_eval --rescore <file> --judge-missing-only`."
            )
    return notes


def case_table(configs: dict[tuple[str, ...], list[Run]]) -> list[str]:
    """Cases that are not PASS in every run of every configuration; cell = passes/runs."""
    order: dict[str, str] = {}
    for runs in configs.values():
        for run in runs:
            for r in run.rows:
                order.setdefault(r["id"], r["category"])

    lines = []
    for case_id, category in order.items():
        cells, always_passed = [], True
        for runs in configs.values():
            verdicts = [final(r) for run in runs for r in run.rows if r["id"] == case_id]
            if not verdicts:
                cells.append("—")
                continue
            passes = verdicts.count(PASS)
            always_passed &= passes == len(verdicts)
            judge_errors = sum(
                r.get("judge_verdict") == JUDGE_ERROR for run in runs for r in run.rows if r["id"] == case_id
            )
            extras = []
            if judge_errors:
                extras.append(f"{judge_errors} judge error" + ("s" if judge_errors != 1 else ""))
            if verdicts.count(ERROR):
                extras.append(f"{verdicts.count(ERROR)} error" + ("s" if verdicts.count(ERROR) != 1 else ""))
            if verdicts.count(NOT_EXERCISED):
                extras.append(f"{verdicts.count(NOT_EXERCISED)} not exercised")
            cells.append(f"{passes}/{len(verdicts)}" + (f" ({', '.join(extras)})" if extras else ""))
        if not always_passed:
            lines.append(f"| {case_id} | {category} | " + " | ".join(cells) + " |")

    if not lines:
        return ["Every case passed in every run of every configuration."]
    header = "| Case | Category | " + " | ".join(config_label(c) for c in configs) + " |"
    return [header, "|---|---|" + "---|" * len(configs), *lines]


def _models(runs: list[Run], field: str) -> str:
    values = sorted({str(r.get(field) or "none") for run in runs for r in run.rows})
    return ", ".join(f"`{v}`" for v in values)


def render(tag: str, runs: list[Run]) -> str:
    grouped = group_runs(runs)
    first, last = runs[0].started, runs[-1].started
    lines = [
        f"# Ablation — tag `{tag}`",
        "",
        f"Model: {_models(runs, 'llm_model')} · judge: {_models(runs, 'judge_model')}",
        "",
        f"{len(runs)} run(s) from {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}."
        " Passes are counted on the Final verdict.",
    ]
    for suite, configs in grouped.items():
        sizes = sorted({len(run.rows) for runs_ in configs.values() for run in runs_})
        size_text = f"{sizes[0]} cases per run" if len(sizes) == 1 else "cases per run vary: " + ", ".join(map(str, sizes))
        lines += ["", f"## `{suite}` ({size_text})", "", *suite_table(configs)]
        notes = incomplete_notes(configs)
        if notes:
            lines += ["", *notes]
        lines += ["", "### Cases not passed in every run", "", *case_table(configs)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reports", type=Path, default=EVALS_DIR / "reports")
    args = parser.parse_args()

    if not TAG_PATTERN.fullmatch(args.tag):
        parser.error("--tag may only contain letters, digits, '.', '_' and '-'")
    runs = load_runs(args.reports, args.tag)
    if not runs:
        parser.error(f"no results in {args.reports} carry the tag {args.tag!r}")

    out = args.reports / f"ablation-{args.tag}.md"
    out.write_text(render(args.tag, runs), encoding="utf-8")
    print(f"{len(runs)} run(s) summarised. Report: {out}")


if __name__ == "__main__":
    main()
