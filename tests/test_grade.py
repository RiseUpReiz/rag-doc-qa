import json

import pytest

from evals import grade, run_eval
from evals.ablation import load_runs
from evals.grade import grade_tag, plan
from evals.judge import JUDGE_ERROR
from evals.scoring import FAIL, PASS
from tests.test_judge_missing import ScriptedJudge, row, write


class NoCallsJudge:
    def invoke(self, prompt):
        raise AssertionError("the judge should not be called")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A reports folder plus a cases file defining every case id the tests use."""
    monkeypatch.setattr(run_eval.settings, "judge_model", "judge-pro")
    reports = tmp_path / "reports"
    reports.mkdir()
    cases = write(tmp_path / "cases.jsonl", [
        {"id": case_id, "category": "cat", "must_contain_any": ["ok"], "expected_behavior": f"rubric {case_id}"}
        for case_id in ["a1", "a2", "a3", "b1", "b2"]
    ])
    return reports, cases


def run_file(reports, stamp, rows, cases, **extra):
    return write(reports / f"results-{stamp}.jsonl", [row(*r, cases_file=str(cases), **extra) for r in rows])


def files(reports):
    return sorted(p.name for p in reports.glob("results-*.jsonl"))


def test_follows_rescore_chains_and_grades_only_the_latest_version(setup):
    reports, cases = setup
    original = run_file(reports, "20261007-100000", [("a1", PASS), ("a2", JUDGE_ERROR), ("a3", JUDGE_ERROR)], cases)
    partial = run_file(reports, "20261007-110000", [("a1", PASS), ("a2", FAIL), ("a3", JUDGE_ERROR)], cases,
                       rescored_from=original.name)

    [pending], problems = plan(reports, "t1")
    assert (pending.run.path, pending.calls, problems) == (partial, 1, [])

    judge = ScriptedJudge(PASS)
    assert grade_tag(reports, "t1", delay=0, llm=judge) == 0

    assert judge.questions == ["question a3"]
    [latest] = load_runs(reports, "t1")
    assert latest.rescored_from == partial.name
    assert [r["judge_verdict"] for r in latest.rows] == [PASS, FAIL, PASS]


def test_second_invocation_does_not_grade_again(setup, capsys):
    reports, cases = setup
    run_file(reports, "20261007-100000", [("a1", JUDGE_ERROR), ("a2", None)], cases)
    assert grade_tag(reports, "t1", delay=0, llm=ScriptedJudge(PASS, FAIL)) == 0
    after_first = files(reports)

    assert grade_tag(reports, "t1", delay=0, llm=NoCallsJudge()) == 0

    assert files(reports) == after_first
    assert "Nothing to grade" in capsys.readouterr().out


def test_resumes_after_a_daily_quota_stop(setup, capsys):
    reports, cases = setup
    run_file(reports, "20261007-100000", [("a1", JUDGE_ERROR), ("a2", JUDGE_ERROR)], cases)
    run_file(reports, "20261007-100100", [("b1", JUDGE_ERROR), ("b2", PASS)], cases, defences=["prompt"])

    first = ScriptedJudge(PASS, None)
    assert grade_tag(reports, "t1", delay=0, llm=first) == 2

    out = capsys.readouterr().out
    assert out.index("3 judge call(s) needed across 2 run(s)") < out.index("[1/2]")
    assert "Judge daily quota exhausted. 2 judge call(s) remain" in out
    assert first.questions == ["question a1", "question a2"]
    assert "[2/2]" not in out

    second = ScriptedJudge(FAIL, PASS)
    assert grade_tag(reports, "t1", delay=0, llm=second) == 0

    assert second.questions == ["question a2", "question b1"]
    runs = load_runs(reports, "t1")
    assert [r.judge_errors for r in runs] == [0, 0]
    assert [[row["judge_verdict"] for row in r.rows] for r in runs] == [[PASS, FAIL], [PASS, PASS]]


def test_changed_rubric_counts_as_a_call_needed(setup, tmp_path):
    reports, cases = setup
    run_file(reports, "20261007-100000", [("a1", PASS), ("a2", PASS)], cases)
    cases.write_text(cases.read_text(encoding="utf-8").replace("rubric a2", "stricter rubric a2"), encoding="utf-8")

    [pending], _ = plan(reports, "t1")
    assert pending.calls == 1


def test_run_with_missing_cases_file_is_reported_and_skipped(setup, tmp_path, capsys):
    reports, cases = setup
    run_file(reports, "20261007-100000", [("a1", JUDGE_ERROR)], tmp_path / "gone.jsonl")

    assert grade_tag(reports, "t1", delay=0, llm=NoCallsJudge()) == 0
    assert "cases file" in capsys.readouterr().out


def test_dry_run_counts_without_judging(setup):
    reports, cases = setup
    run_file(reports, "20261007-100000", [("a1", JUDGE_ERROR), ("a2", None), ("a3", PASS)], cases)
    before = files(reports)

    assert grade_tag(reports, "t1", delay=0, llm=NoCallsJudge(), dry_run=True) == 2
    assert files(reports) == before


def test_cli_rejects_unknown_tag(setup, monkeypatch):
    reports, _ = setup
    monkeypatch.setattr("sys.argv", ["grade", "--tag", "nope", "--reports", str(reports)])
    with pytest.raises(SystemExit):
        grade.main()
