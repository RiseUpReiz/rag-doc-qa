import json

from evals import run_eval
from evals.scoring import ERROR, FAIL, PASS

SAVED_RESULTS = [
    {
        "id": "u3",
        "category": "unanswerable",
        "question": "Who is the CEO of Halden Marine Systems?",
        "expect_refusal": True,
        "answer": "Based on the provided documents, there is no mention of who the CEO of "
        "Halden Marine Systems is.",
        "status": FAIL,
        "reasons": ["expected a refusal but got an answer"],
    },
    {
        "id": "g1",
        "category": "grounded",
        "question": "How many days of annual leave?",
        "must_contain_any": ["22"],
        "answer": "You get 18 days.",
        "status": PASS,
        "reasons": [],
    },
    {
        "id": "g2",
        "category": "grounded",
        "question": "What is the daily meal allowance?",
        "must_contain_any": ["60"],
        "answer": "",
        "status": ERROR,
        "reasons": ["not run: daily quota exhausted"],
    },
]


def test_rescore_regrades_saved_answers_without_model_calls(tmp_path, monkeypatch):
    def no_model_calls(*args, **kwargs):
        raise AssertionError("rescore must not call the model")

    monkeypatch.setattr(run_eval, "answer_question", no_model_calls)
    monkeypatch.setattr(run_eval, "build_eval_store", no_model_calls)

    results_path = tmp_path / "results-20261006-101319.jsonl"
    results_path.write_text(
        "".join(json.dumps(r) + "\n" for r in SAVED_RESULTS), encoding="utf-8"
    )
    out_dir = tmp_path / "reports"

    report_path, results = run_eval.rescore_file(results_path, out_dir)

    by_id = {r["id"]: r for r in results}
    assert by_id["u3"]["status"] == PASS
    assert by_id["u3"]["reasons"] == []
    assert by_id["g1"]["status"] == FAIL
    assert by_id["g2"]["status"] == ERROR
    assert by_id["g2"]["reasons"] == ["not run: daily quota exhausted"]

    report = report_path.read_text(encoding="utf-8")
    assert "Rescored from results-20261006-101319.jsonl (no new model calls)" in report
    assert "1/2 scored passed · 1 errors (not run)" in report
