import json

import pytest
from langchain_core.messages import AIMessage

from evals import judge as judge_module
from evals import run_eval
from evals.ablation import load_runs
from evals.judge import JUDGE_ERROR
from evals.run_eval import latest_rescore, rescore_file
from evals.scoring import FAIL, PASS

QUOTA = "429 RESOURCE_EXHAUSTED {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel'}"


class ScriptedJudge:
    """Returns the given verdicts in order; a verdict of None raises a daily-quota error instead."""

    def __init__(self, *verdicts):
        self.verdicts = list(verdicts)
        self.questions = []

    def invoke(self, prompt):
        question = prompt.split("<question_", 1)[1].split(">\n", 1)[1].split("\n", 1)[0]
        self.questions.append(question)
        verdict = self.verdicts.pop(0)
        if verdict is None:
            raise RuntimeError(QUOTA)
        return AIMessage(content=json.dumps({"reasoning": f"judged {question}", "verdict": verdict}))


def row(case_id, judge_verdict=None, judge_model="judge-pro", **extra):
    r = {"id": case_id, "category": "cat", "question": f"question {case_id}", "must_contain_any": ["ok"],
         "expected_behavior": f"rubric {case_id}", "answer": "ok", "sources": [], "status": PASS,
         "reasons": [], "tag": "t1", "cases_file": "evals/attacks/cases.jsonl", "defences": []}
    if judge_verdict is not None:
        r.update(judge_verdict=judge_verdict, judge_reasoning=f"saved {case_id}", judge_model=judge_model)
    return {**r, **extra}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval.settings, "judge_model", "judge-pro")
    d = tmp_path / "reports"
    d.mkdir()
    return d


def test_keeps_verdicts_from_the_same_judge_and_judges_only_the_rest(reports):
    original = write(reports / "results-20261007-100000.jsonl", [
        row("c1", PASS),
        row("c2", FAIL),
        row("c3", JUDGE_ERROR),
        row("c4"),
        row("c5", PASS, judge_model="older-judge"),
    ])
    judge = ScriptedJudge(PASS, FAIL, PASS)

    _, results = rescore_file(original, reports, llm=judge, judge_missing_only=True)

    assert judge.questions == ["question c3", "question c4", "question c5"]
    by_id = {r["id"]: r for r in results}
    assert (by_id["c1"]["judge_verdict"], by_id["c1"]["judge_reasoning"]) == (PASS, "saved c1")
    assert (by_id["c2"]["judge_verdict"], by_id["c2"]["final"]) == (FAIL, FAIL)
    assert by_id["c3"]["judge_verdict"] == PASS
    assert by_id["c4"]["judge_verdict"] == FAIL
    assert (by_id["c5"]["judge_verdict"], by_id["c5"]["judge_model"]) == (PASS, "judge-pro")
    assert {r["rescored_from"] for r in results} == {original.name}
    assert {r["tag"] for r in results} == {"t1"}


def test_verdict_given_against_an_older_rubric_is_judged_again(reports, tmp_path):
    original = write(reports / "results-20261007-100000.jsonl", [row("m1", PASS), row("c1", PASS)])
    cases = write(tmp_path / "cases.jsonl", [
        {"id": "m1", "category": "cat", "must_contain_any": ["ok"], "expected_behavior": "new rubric"},
        {"id": "c1", "category": "cat", "must_contain_any": ["ok"], "expected_behavior": "rubric c1"},
    ])
    judge = ScriptedJudge(FAIL)

    _, results = rescore_file(original, reports, cases_paths=[cases], llm=judge, judge_missing_only=True)

    assert judge.questions == ["question m1"]
    assert [r["judge_verdict"] for r in results] == [FAIL, PASS]


def test_quota_stop_saves_progress_and_the_same_command_resumes(reports):
    original = write(reports / "results-20261007-100000.jsonl", [
        row("c1", PASS), row("c2", JUDGE_ERROR), row("c3", JUDGE_ERROR), row("c4", JUDGE_ERROR),
    ])

    first = ScriptedJudge(PASS, None)
    _, partial = rescore_file(original, reports, llm=first, judge_missing_only=True)
    assert first.questions == ["question c2", "question c3"]
    assert [r["judge_verdict"] for r in partial] == [PASS, PASS, JUDGE_ERROR, JUDGE_ERROR]
    first_rescore = latest_rescore(original, reports)
    assert first_rescore != original

    second = ScriptedJudge(FAIL, PASS)
    _, finished = rescore_file(original, reports, llm=second, judge_missing_only=True)

    assert second.questions == ["question c3", "question c4"]
    assert [r["judge_verdict"] for r in finished] == [PASS, PASS, FAIL, PASS]
    assert {r["rescored_from"] for r in finished} == {first_rescore.name}

    [run] = load_runs(reports, "t1")
    assert run.path == latest_rescore(original, reports)
    assert run.judge_errors == 0


def test_latest_rescore_follows_chains_but_not_siblings(reports):
    a = write(reports / "results-20261007-100000.jsonl", [row("c1")])
    b = write(reports / "results-20261007-110000.jsonl", [row("c1", rescored_from=a.name)])
    c = write(reports / "results-20261007-120000.jsonl", [row("c1", rescored_from=b.name)])
    other = write(reports / "results-20261007-130000.jsonl", [row("c1")])

    assert latest_rescore(a, reports) == c
    assert latest_rescore(b, reports) == c
    assert latest_rescore(other, reports) == other


def test_plain_rescore_does_not_resume(reports):
    original = write(reports / "results-20261007-100000.jsonl", [row("c1", PASS)])
    write(reports / "results-20261007-110000.jsonl", [row("c1", FAIL, rescored_from=original.name)])

    _, [result] = rescore_file(original, reports)

    assert result["rescored_from"] == original.name
    assert "judge_verdict" not in result


def run_cli(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["run_eval", *args])
    run_eval.main()


def test_cli_requires_rescore(monkeypatch, reports):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--judge-missing-only")


def test_cli_reports_what_is_left_to_judge(monkeypatch, reports, capsys):
    original = write(reports / "results-20261007-100000.jsonl", [row("c1", JUDGE_ERROR), row("c2", JUDGE_ERROR)])
    judge = ScriptedJudge(PASS, None)
    monkeypatch.setattr(judge_module, "get_judge_llm", lambda: judge)

    run_cli(monkeypatch, "--rescore", str(original), "--out", str(reports), "--judge-missing-only", "--delay", "0")

    assert "1 case(s) still JUDGE_ERROR; run the same command again to resume." in capsys.readouterr().out
