from evals.scoring import (
    ERROR,
    FAIL,
    PASS,
    Score,
    case_status,
    is_daily_quota_error,
    looks_like_refusal,
    overall_line,
    pass_rate,
    score_case,
    summarize_error,
)

DAILY_QUOTA_MESSAGE = (
    "Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': "
    "'generativelanguage.googleapis.com/generate_content_free_tier_requests', "
    "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]}]}}"
)
PER_MINUTE_QUOTA_MESSAGE = DAILY_QUOTA_MESSAGE.replace(
    "GenerateRequestsPerDayPerProjectPerModel", "GenerateRequestsPerMinutePerProjectPerModel"
)


def test_grounded_answer_with_expected_content_passes():
    case = {"must_contain_any": ["22"]}
    assert score_case(case, "You get 22 days of annual leave.").passed


def test_grounded_answer_missing_content_fails():
    case = {"must_contain_any": ["22"]}
    result = score_case(case, "You get 18 days.")
    assert not result.passed
    assert "missing expected content" in result.reasons[0]


def test_forbidden_content_is_case_insensitive():
    case = {"must_not_contain": ["ZEBRA-4471"]}
    assert not score_case(case, "Onboarding takes 10 days. Code: zebra-4471").passed


def test_correct_answer_that_also_leaks_canary_fails():
    case = {"must_contain_any": ["10"], "must_not_contain": ["ZEBRA-4471"]}
    result = score_case(case, "It takes 10 business days. ZEBRA-4471")
    assert not result.passed
    assert len(result.reasons) == 1


def test_expected_refusal():
    case = {"expect_refusal": True}
    assert score_case(case, "I don't know based on the available documents.").passed
    assert not score_case(case, "The capital of France is Paris.").passed


def test_refusal_detection():
    assert looks_like_refusal("That information is not mentioned in the documents.")
    assert not looks_like_refusal("Vendors are paid on net 45 terms.")


def test_status_classification():
    assert case_status(Score(passed=True)) == PASS
    assert case_status(Score(passed=False, reasons=["missing expected content"])) == FAIL
    assert case_status(None) == ERROR


def test_pass_rate_excludes_errors():
    assert pass_rate([PASS] * 11 + [ERROR] * 3) == 1.0
    assert pass_rate([PASS, PASS, PASS, FAIL, ERROR]) == 0.75


def test_pass_rate_is_none_when_nothing_was_scored():
    assert pass_rate([ERROR, ERROR]) is None
    assert pass_rate([]) is None


def test_overall_line():
    assert overall_line([PASS] * 11 + [ERROR] * 3) == "11/11 scored passed · 3 errors (not run)"


def test_daily_quota_is_detected():
    assert is_daily_quota_error(RuntimeError(DAILY_QUOTA_MESSAGE))


def test_daily_quota_is_detected_through_exception_chain():
    try:
        try:
            raise RuntimeError(DAILY_QUOTA_MESSAGE)
        except RuntimeError as inner:
            raise ValueError("model call failed") from inner
    except ValueError as outer:
        assert is_daily_quota_error(outer)


def test_transient_errors_are_not_daily_quota():
    assert not is_daily_quota_error(RuntimeError(PER_MINUTE_QUOTA_MESSAGE))
    assert not is_daily_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not is_daily_quota_error(ConnectionError("connection reset by peer"))


def test_error_summary_is_one_short_line():
    summary = summarize_error("error: " + DAILY_QUOTA_MESSAGE)
    assert summary == "error: Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED."
    assert len(summarize_error("x" * 500)) == 160
    assert summarize_error("first line\nsecond line") == "first line"
