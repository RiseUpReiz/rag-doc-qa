import json

import pytest

from evals import ablation, run_eval
from evals.ablation import case_table, group_runs, load_runs, render, suite_table
from evals.run_eval import NOT_EXERCISED
from evals.scoring import ERROR, FAIL, PASS

ATTACKS = "evals/attacks/cases.jsonl"
BASELINE = "evals/cases.jsonl"


def write_run(reports, stamp, finals, defences=(), cases_file=ATTACKS, tag="t1", **extra):
    """A fake results file: finals maps case id -> Final verdict."""
    rows = [
        {"id": case_id, "category": "cat_" + case_id, "question": "q", "answer": "a", "sources": [],
         "status": verdict, "final": verdict, "reasons": [], "tag": tag, "cases_file": cases_file,
         "defences": list(defences), "llm_model": "gemini-3.6-flash", "judge_model": "judge-pro", **extra}
        for case_id, verdict in finals.items()
    ]
    path = reports / f"results-{stamp}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def reports(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return d


# --- loading and grouping ---

def test_only_runs_with_the_tag_are_loaded(reports):
    write_run(reports, "20261007-100000", {"c1": PASS})
    write_run(reports, "20261007-100100", {"c1": PASS}, tag="other")
    write_run(reports, "20261007-100200", {"c1": PASS}, tag=None)
    (reports / "results-20261007-100300.jsonl").write_text(
        json.dumps({"id": "c1", "category": "x", "status": PASS}) + "\n", encoding="utf-8"
    )

    runs = load_runs(reports, "t1")
    assert [r.path.name for r in runs] == ["results-20261007-100000.jsonl"]


def test_groups_by_suite_and_defences_in_ablation_order(reports):
    write_run(reports, "20261007-100000", {"c1": PASS}, defences=["links", "trust", "prompt"])
    write_run(reports, "20261007-100100", {"c1": PASS}, defences=[])
    write_run(reports, "20261007-100200", {"c1": PASS}, defences=["trust", "prompt"])
    write_run(reports, "20261007-100300", {"c1": PASS}, defences=["prompt"])
    write_run(reports, "20261007-100400", {"c1": PASS}, defences=[])
    write_run(reports, "20261007-100500", {"g1": PASS}, defences=["prompt"], cases_file=BASELINE)
    write_run(reports, "20261007-100600", {"c1": PASS}, defences=["links"])

    grouped = group_runs(load_runs(reports, "t1"))

    assert list(grouped) == [ATTACKS, BASELINE]
    assert list(grouped[ATTACKS]) == [(), ("prompt",), ("prompt", "trust"), ("prompt", "trust", "links"), ("links",)]
    assert len(grouped[ATTACKS][()]) == 2
    assert list(grouped[BASELINE]) == [("prompt",)]


def test_file_that_was_rescored_is_ignored(reports):
    write_run(reports, "20261007-100000", {"c1": FAIL})
    write_run(reports, "20261007-110000", {"c1": PASS})
    write_run(reports, "20261007-120000", {"c1": PASS}, rescored_from="results-20261007-100000.jsonl")

    runs = load_runs(reports, "t1")

    assert [r.path.name for r in runs] == ["results-20261007-110000.jsonl", "results-20261007-120000.jsonl"]


def test_chain_of_rescores_counts_the_run_once_with_its_latest_grading(reports):
    write_run(reports, "20261007-100000", {"c1": FAIL})
    write_run(reports, "20261007-120000", {"c1": FAIL}, rescored_from="results-20261007-100000.jsonl")
    write_run(reports, "20261007-130000", {"c1": PASS}, rescored_from="results-20261007-120000.jsonl")

    assert [r.path.name for r in load_runs(reports, "t1")] == ["results-20261007-130000.jsonl"]


def test_two_separate_rescores_of_one_run_keep_only_the_newest(reports):
    write_run(reports, "20261007-100000", {"c1": FAIL})
    write_run(reports, "20261007-120000", {"c1": PASS}, rescored_from="results-20261007-100000.jsonl")
    write_run(reports, "20261007-130000", {"c1": FAIL}, rescored_from="results-20261007-100000.jsonl")

    assert [r.path.name for r in load_runs(reports, "t1")] == ["results-20261007-130000.jsonl"]


def test_run_mixing_configurations_is_rejected(reports):
    path = write_run(reports, "20261007-100000", {"c1": PASS})
    rows = [json.loads(line) for line in path.open()]
    rows.append({**rows[0], "id": "c2", "defences": ["prompt"]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="mixes"):
        group_runs(load_runs(reports, "t1"))


# --- suite table ---

def test_suite_table_lists_passes_per_run_and_range(reports):
    for i, finals in enumerate([{"c1": PASS, "c2": PASS}, {"c1": PASS, "c2": FAIL}, {"c1": PASS, "c2": PASS}]):
        write_run(reports, f"20261007-10000{i}", finals)
    write_run(reports, "20261007-110000", {"c1": PASS, "c2": PASS}, defences=["prompt"])

    configs = group_runs(load_runs(reports, "t1"))[ATTACKS]
    lines = suite_table(configs)

    assert "| none | 3 | 2, 1, 2 | 1–2 | 0 | 0 | 0 |" in lines
    assert "| prompt | 1 | 2 | 2–2 | 0 | 0 | 0 |" in lines


def test_errors_and_not_exercised_are_counted_and_flagged(reports):
    write_run(reports, "20261007-100000", {"c1": PASS, "k1": NOT_EXERCISED})
    write_run(reports, "20261007-100100", {"c1": ERROR, "k1": PASS})

    report = render("t1", load_runs(reports, "t1"))

    assert "| none ⚠ | 2 | 1, 1 | 1–1 | 1 | 1 | 0 |" in report
    assert "⚠ **none**: 1 ERROR and 1 NOT_EXERCISED result(s) across 2 run(s)" in report


def test_unequal_run_sizes_show_passed_out_of_total(reports):
    write_run(reports, "20261007-100000", {"c1": PASS, "c2": PASS})
    write_run(reports, "20261007-100100", {"c1": PASS})

    lines = suite_table(group_runs(load_runs(reports, "t1"))[ATTACKS])
    assert "| none | 2 | 2/2, 1/1 | 1–2 | 0 | 0 | 0 |" in lines


def test_runs_with_judge_errors_are_flagged_and_left_out_of_min_max(reports):
    write_run(reports, "20261007-100000", {"c1": PASS, "c2": PASS})
    write_run(reports, "20261007-100100", {"c1": PASS, "c2": FAIL})
    write_run(reports, "20261007-100200", {"c1": PASS, "c2": PASS}, judge_verdict="JUDGE_ERROR")

    report = render("t1", load_runs(reports, "t1"))

    assert "| none ⚠ | 3 | 2, 1, 2 (incomplete) | 1–2 | 0 | 0 | 2 |" in report
    assert "1 of 3 run(s) still have JUDGE_ERROR rows: `results-20261007-100200.jsonl` (2)" in report
    assert "| c2 | cat_c2 | 2/3 (1 judge error) |" in report


def test_min_max_is_blank_when_every_run_is_incomplete(reports):
    write_run(reports, "20261007-100000", {"c1": PASS}, judge_verdict="JUDGE_ERROR")
    lines = suite_table(group_runs(load_runs(reports, "t1"))[ATTACKS])
    assert "| none ⚠ | 1 | 1 (incomplete) | — | 0 | 0 | 1 |" in lines


# --- per-case table ---

def test_case_table_shows_only_cases_not_always_passed(reports):
    write_run(reports, "20261007-100000", {"c1": PASS, "p1": FAIL, "m1": PASS})
    write_run(reports, "20261007-100100", {"c1": PASS, "p1": FAIL, "m1": FAIL})
    write_run(reports, "20261007-100200", {"c1": PASS, "p1": PASS, "m1": PASS}, defences=["prompt"])
    write_run(reports, "20261007-100300", {"c1": PASS, "p1": PASS, "m1": PASS}, defences=["prompt"])

    lines = case_table(group_runs(load_runs(reports, "t1"))[ATTACKS])

    assert lines[0] == "| Case | Category | none | prompt |"
    assert "| p1 | cat_p1 | 0/2 | 2/2 |" in lines
    assert "| m1 | cat_m1 | 1/2 | 2/2 |" in lines
    assert not any(line.startswith("| c1 ") for line in lines)


def test_case_table_marks_errors_and_missing_cases(reports):
    write_run(reports, "20261007-100000", {"c1": PASS, "k1": ERROR})
    write_run(reports, "20261007-100100", {"c1": PASS, "k1": NOT_EXERCISED})
    write_run(reports, "20261007-100200", {"c1": FAIL}, defences=["prompt"])

    lines = case_table(group_runs(load_runs(reports, "t1"))[ATTACKS])

    assert "| k1 | cat_k1 | 0/2 (1 error, 1 not exercised) | — |" in lines
    assert "| c1 | cat_c1 | 2/2 | 0/1 |" in lines


def test_case_table_when_everything_passed(reports):
    write_run(reports, "20261007-100000", {"c1": PASS})
    assert case_table(group_runs(load_runs(reports, "t1"))[ATTACKS]) == [
        "Every case passed in every run of every configuration."
    ]


# --- report and CLI ---

def test_report_header_and_file(reports, monkeypatch):
    write_run(reports, "20261007-093000", {"c1": PASS})
    write_run(reports, "20261008-151500", {"g1": FAIL}, cases_file=BASELINE, judge_model=None)
    monkeypatch.setattr("sys.argv", ["ablation", "--tag", "t1", "--reports", str(reports)])

    ablation.main()

    report = (reports / "ablation-t1.md").read_text(encoding="utf-8")
    assert report.startswith("# Ablation — tag `t1`")
    assert "Model: `gemini-3.6-flash` · judge: `judge-pro`, `none`" in report
    assert "2 run(s) from 2026-10-07 09:30 to 2026-10-08 15:15." in report
    assert f"## `{ATTACKS}` (1 cases per run)" in report
    assert f"## `{BASELINE}` (1 cases per run)" in report


def test_cli_errors_when_no_runs_have_the_tag(reports, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ablation", "--tag", "missing", "--reports", str(reports)])
    with pytest.raises(SystemExit):
        ablation.main()


# --- run_eval --repeat / --tag ---

def run_main(tmp_path, monkeypatch, *extra_args, answers=None):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps({"id": "c1", "category": "control", "question": "q",
                                      "must_contain_any": ["22"]}) + "\n", encoding="utf-8")
    built, answers = [], list(answers or [])
    monkeypatch.setattr(run_eval, "build_eval_store", lambda corpus, manifest=None: built.append(corpus))

    def fake_answer(question, store=None, defences=None, allowed_domains=None):
        answer = answers.pop(0) if answers else "22 days"
        if isinstance(answer, Exception):
            raise answer
        return {"answer": answer, "sources": []}

    monkeypatch.setattr(run_eval, "answer_question", fake_answer)
    monkeypatch.setattr("sys.argv", ["run_eval", "--cases", str(cases_path), "--corpus", str(tmp_path),
                                     "--out", str(tmp_path / "reports"), "--delay", "0", *extra_args])
    run_eval.main()
    return built


def test_repeat_writes_one_pair_per_run_and_tags_every_row(tmp_path, monkeypatch):
    built = run_main(tmp_path, monkeypatch, "--repeat", "3", "--tag", "stage4", "--defences", "prompt",
                     answers=["22 days", "no idea", "22 days"])

    assert len(built) == 1
    results_files = sorted((tmp_path / "reports").glob("results-*.jsonl"))
    assert len(results_files) == 3 and len(list((tmp_path / "reports").glob("report-*.md"))) == 3
    rows = [json.loads(line) for f in results_files for line in f.open()]
    assert {(r["tag"], r["cases_file"], tuple(r["defences"])) for r in rows} == {
        ("stage4", (tmp_path / "cases.jsonl").as_posix(), ("prompt",))
    }
    assert sorted(r["final"] for r in rows) == [FAIL, PASS, PASS]

    report = render("stage4", load_runs(tmp_path / "reports", "stage4"))
    assert "| prompt | 3 | 1, 0, 1 | 0–1 | 0 | 0 |" in report
    assert "| c1 | control | 2/3 |" in report


def test_repeat_stops_when_the_daily_quota_runs_out(tmp_path, monkeypatch):
    quota = RuntimeError("429 RESOURCE_EXHAUSTED {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel'}")
    run_main(tmp_path, monkeypatch, "--repeat", "3", "--tag", "t", answers=[quota])
    assert len(list((tmp_path / "reports").glob("results-*.jsonl"))) == 1


@pytest.mark.parametrize("tag", ["has space", "a/b", "x;y"])
def test_bad_tags_are_rejected(tag, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(tmp_path, monkeypatch, "--tag", tag)


def test_rescore_keeps_tag_and_records_rescored_from(tmp_path, reports):
    original = write_run(reports, "20261007-100000", {"c1": PASS})
    _, results = run_eval.rescore_file(original, reports)
    assert results[0]["tag"] == "t1"
    assert results[0]["rescored_from"] == "results-20261007-100000.jsonl"
    [run] = load_runs(reports, "t1")
    assert run.path.name != original.name


def test_unique_stamp_avoids_overwriting(tmp_path, monkeypatch):
    class FixedDatetime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2026, 10, 7, 10, 0, 0)

    monkeypatch.setattr(run_eval, "datetime", FixedDatetime)
    (tmp_path / "results-20261007-100000.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "results-20261007-100000-2.jsonl").write_text("", encoding="utf-8")
    assert run_eval.unique_stamp(tmp_path) == "20261007-100000-3"
