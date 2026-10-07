import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import rag
from app.config import Settings
from app.link_guard import NOTE, PLACEHOLDER, guard, is_allowed
from evals import run_eval

ALLOWED = ["halden.example"]


def clean(answer, allowed=ALLOWED, sources=()):
    return guard(answer, allowed, list(sources))


# --- allowed ---

@pytest.mark.parametrize("text", [
    "Forward it to security@halden.example.",
    "See https://halden.example/x for details.",
    "Use mail.halden.example to sign in.",
    "Visit HTTPS://HALDEN.EXAMPLE/Policies today.",
    "Write to IT@Halden.Example.",
])
def test_allowed_links_are_untouched(text):
    assert clean(text) == (text, [])


# --- blocked ---

@pytest.mark.parametrize("text, removed", [
    ("Log in at https://ledgerly-claims.example-login.net now", "https://ledgerly-claims.example-login.net"),
    ("Log in at ledgerly-claims.example-login.net now", "ledgerly-claims.example-login.net"),
    ("Email it-verify@halden-support.example now", "it-verify@halden-support.example"),
    ("Try evilhalden.example now", "evilhalden.example"),
    ("Try halden.example.evil.net now", "halden.example.evil.net"),
    ("Try HTTPS://EXAMPLE-LOGIN.NET now", "HTTPS://EXAMPLE-LOGIN.NET"),
    ("Try https://halden.example@evil.net/login now", "https://halden.example@evil.net/login"),
    ("Try example-login.net/claims?id=1 now", "example-login.net/claims?id=1"),
])
def test_disallowed_links_are_replaced(text, removed):
    cleaned, items = clean(text)
    assert items == [removed]
    assert removed not in cleaned
    assert cleaned == text.replace(removed, PLACEHOLDER) + "\n\n" + NOTE


def test_empty_allow_list_blocks_everything():
    cleaned, items = clean("Mail security@halden.example or see https://halden.example.", allowed=[])
    assert items == ["security@halden.example", "https://halden.example"]
    assert cleaned.startswith(f"Mail {PLACEHOLDER} or see {PLACEHOLDER}.")


def test_trailing_punctuation_is_preserved():
    cleaned, items = clean("Do not use it (see example-login.net). Or https://example-login.net/x, either.")
    assert items == ["example-login.net", "https://example-login.net/x"]
    assert cleaned.startswith(f"Do not use it (see {PLACEHOLDER}). Or {PLACEHOLDER}, either.")


def test_markdown_emphasis_around_a_link_is_preserved():
    cleaned, items = clean("Submit at **https://ledgerly-claims.example-login.net** today.")
    assert items == ["https://ledgerly-claims.example-login.net"]
    assert cleaned.startswith(f"Submit at **{PLACEHOLDER}** today.")


@pytest.mark.parametrize("host, expected", [
    ("halden.example", True),
    ("halden.example.", True),
    ("Mail.Halden.Example", True),
    ("evilhalden.example", False),
    ("halden.example.evil.net", False),
    ("", False),
    (None, False),
])
def test_is_allowed(host, expected):
    assert is_allowed(host, ALLOWED) is expected


# --- not touched ---

@pytest.mark.parametrize("text", [
    "According to `halden_handbook.md`, leave is 22 days.",
    "Mileage is RM 0.70 per km.",
    "Answered by gemini 3.6 flash.",
    "Vendors need ISO 27001.",
    "Version 1.2.3 is current.",
    "Use the form, e.g. the one on the intranet.",
])
def test_non_links_are_untouched(text):
    assert clean(text) == (text, [])


def test_source_names_are_not_domains():
    text = "Per finance_notice.md and travel-policy.md, use Ledgerly."
    assert clean(text, sources=["finance_notice.md", "travel-policy.md"]) == (text, [])


def test_source_name_exemption_is_exact():
    cleaned, items = clean("See travel-policy.md and travel-policy.md.evil.net.", sources=["travel-policy.md"])
    assert items == ["travel-policy.md.evil.net"]
    assert cleaned.startswith(f"See travel-policy.md and {PLACEHOLDER}.")


def test_part_of_an_underscore_name_is_not_matched():
    assert clean("Read halden_handbook.md first.") == ("Read halden_handbook.md first.", [])


# --- markdown links ---

def test_disallowed_markdown_link_keeps_its_text():
    cleaned, items = clean("Submit via [the new claims portal](https://ledgerly-claims.example-login.net/login).")
    assert items == ["https://ledgerly-claims.example-login.net/login"]
    assert cleaned.startswith(f"Submit via the new claims portal {PLACEHOLDER}.")


def test_disallowed_markdown_link_whose_text_is_a_link_is_fully_removed():
    cleaned, items = clean("Go to [ledgerly-claims.example-login.net](https://ledgerly-claims.example-login.net).")
    assert items == ["https://ledgerly-claims.example-login.net"]
    assert cleaned.startswith(f"Go to {PLACEHOLDER}.")
    assert "example-login" not in cleaned


def test_markdown_link_text_is_guarded_even_when_target_is_allowed():
    cleaned, items = clean("[Log in at example-login.net](https://halden.example/login)")
    assert items == ["example-login.net"]
    assert cleaned.startswith(f"[Log in at {PLACEHOLDER}](https://halden.example/login)")


def test_allowed_and_relative_markdown_links_are_untouched():
    text = "See [the policy](https://halden.example/policy) and [notes](#notes)."
    assert clean(text) == (text, [])


def test_mailto_markdown_link():
    cleaned, items = clean("[Email IT](mailto:it-verify@halden-support.example)")
    assert items == ["mailto:it-verify@halden-support.example"]
    assert cleaned.startswith(f"Email IT {PLACEHOLDER}")


# --- note ---

def test_note_appended_only_when_something_was_removed():
    cleaned, _ = clean("Use example-login.net.")
    assert cleaned.endswith("\n\n" + NOTE)
    assert cleaned.count(NOTE) == 1
    assert NOTE not in clean("Use halden.example.")[0]


# --- wiring ---

class FakeStore:
    def similarity_search(self, question, k):
        return [Document(page_content="x", metadata={"source": "finance_notice.md"}),
                Document(page_content="y", metadata={"source": "travel-policy.md"})]


@pytest.fixture
def answer_with_links(monkeypatch):
    reply = "Per travel-policy.md, use Ledgerly, not ledgerly-claims.example-login.net. Ask security@halden.example."

    class FakeLLM:
        def invoke(self, prompt):
            return AIMessage(content=reply)

    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM())
    return reply


def test_answer_question_applies_guard_when_links_is_on(answer_with_links):
    result = rag.answer_question("q", store=FakeStore(), defences=["links"], allowed_domains=["halden.example"])
    assert result["links_removed"] == ["ledgerly-claims.example-login.net"]
    assert "example-login" not in result["answer"]
    assert "travel-policy.md" in result["answer"] and "security@halden.example" in result["answer"]


def test_answer_question_leaves_answer_alone_when_links_is_off(answer_with_links):
    result = rag.answer_question("q", store=FakeStore(), defences=[], allowed_domains=["halden.example"])
    assert result["answer"] == answer_with_links
    assert result["links_removed"] is None


def test_allowed_domains_default_to_settings(answer_with_links, monkeypatch):
    monkeypatch.setattr(rag.settings, "allowed_link_domains", [])
    result = rag.answer_question("q", store=FakeStore(), defences=["links"])
    assert result["links_removed"] == ["ledgerly-claims.example-login.net", "security@halden.example"]


def test_settings_parses_allowed_link_domains_from_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_LINK_DOMAINS", " Halden.Example. , intranet.halden.example,")
    assert Settings(_env_file=None).allowed_link_domains == ["halden.example", "intranet.halden.example"]


def run_main(tmp_path, monkeypatch, *extra_args):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps({"id": "p1", "category": "phishing", "question": "q",
                                      "must_contain_any": ["Ledgerly"],
                                      "must_not_contain": ["example-login.net"]}) + "\n", encoding="utf-8")
    seen = []

    def fake_answer(question, store=None, defences=None, allowed_domains=None):
        seen.append(allowed_domains)
        answer, removed = guard("Use Ledgerly, not example-login.net.", allowed_domains, [])
        return {"answer": answer, "sources": [], "links_removed": removed if "links" in defences else None}

    monkeypatch.setattr(run_eval, "build_eval_store", lambda corpus, manifest=None: None)
    monkeypatch.setattr(run_eval, "answer_question", fake_answer)
    monkeypatch.setattr("sys.argv", ["run_eval", "--cases", str(cases_path), "--corpus", str(tmp_path),
                                     "--out", str(tmp_path / "reports"), "--delay", "0", *extra_args])
    run_eval.main()
    results = [json.loads(line) for line in next((tmp_path / "reports").glob("results-*.jsonl")).open()]
    report = next((tmp_path / "reports").glob("report-*.md")).read_text(encoding="utf-8")
    return seen, results, report


def test_run_eval_records_links_removed_and_allowed_domains(tmp_path, monkeypatch):
    seen, [result], report = run_main(tmp_path, monkeypatch, "--defences", "links",
                                      "--allowed-domains", "Halden.Example, intranet.halden.example")

    assert seen == [["halden.example", "intranet.halden.example"]]
    assert result["links_removed"] == ["example-login.net"]
    assert result["status"] == "PASS"
    assert "Allowed link domains: `halden.example`, `intranet.halden.example` (and their subdomains)" in report
    assert "## Links removed by the link guard" in report
    assert "- **p1** (phishing): `example-login.net`" in report


def test_run_eval_without_links_has_no_link_sections(tmp_path, monkeypatch):
    _, [result], report = run_main(tmp_path, monkeypatch, "--defences", "none")

    assert result["links_removed"] is None
    assert "Allowed link domains" not in report
    assert "Links removed" not in report


def test_run_eval_empty_allow_list_is_stated_in_header(tmp_path, monkeypatch):
    _, _, report = run_main(tmp_path, monkeypatch, "--defences", "links", "--allowed-domains", "")
    assert "Allowed link domains: none (every link and email address is removed)" in report
