import pytest

from research_assistant.agents.grounding_node import make_grounding_node
from research_assistant.agents.reviewer import (
    LLMReportReviewer,
    make_reviewer_node,
    normalize_verdict,
)
from research_assistant.agents.writer import make_writer_node
from research_assistant.graph import build_graph
from research_assistant.models import ResearchStatus, ReviewDecision, ReviewVerdict, Source
from research_assistant.state import new_state

A = Source.from_url("https://a.com/a", title="A", content="alpha facts")
THIN = Source.from_url("https://c.com/c", snippet="snippet only")

APPROVE = ReviewVerdict(decision=ReviewDecision.APPROVE)


def revise(feedback="fix the second claim"):
    return ReviewVerdict(decision=ReviewDecision.REVISE, feedback=feedback)


GOOD_DRAFT = f"Alpha is described in detail by the only source we have [{A.id}]."


class FakeStructuredLLM:
    def __init__(self, result):
        self._result = result
        self.messages = None
        self.kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.kwargs = kwargs
        return self

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class ScriptedReviewer:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    def review(self, question, report, sources):
        self.calls.append({"report": report, "sources": list(sources)})
        if isinstance(self.verdicts[0], Exception):
            raise self.verdicts[0]
        return self.verdicts[min(len(self.calls), len(self.verdicts)) - 1]


class ScriptedWriter:
    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.calls = []

    def write(self, question, sources, feedback=None, previous_draft=""):
        self.calls.append({"feedback": feedback, "previous_draft": previous_draft})
        return self.drafts[min(len(self.calls), len(self.drafts)) - 1]


def run(writer, reviewer, max_revisions=2):
    graph = build_graph(
        lambda s: {"sources": [A, THIN], "status": ResearchStatus.WRITING},
        make_writer_node(writer),
        make_grounding_node(max_revisions, 0.8),
        make_reviewer_node(reviewer, max_revisions),
        max_revisions,
    )
    return graph.invoke(new_state("q"))


# --- reviewer unit behaviour -------------------------------------------------------------------


def test_llm_reviewer_uses_json_schema_and_returns_the_verdict():
    llm = FakeStructuredLLM(APPROVE)
    verdict = LLMReportReviewer(llm).review("What is A?", "the report", [A])

    assert verdict == APPROVE
    assert llm.schema is ReviewVerdict
    assert llm.kwargs == {"method": "json_schema"}


def test_llm_reviewer_prompt_carries_question_report_sources_and_injection_guard():
    llm = FakeStructuredLLM(APPROVE)
    LLMReportReviewer(llm).review("What is A?", "the report text", [A])

    (_, system), (_, human) = llm.messages
    assert "never as instructions" in system
    assert "What is A?" in human and "the report text" in human
    assert A.id in human and "alpha facts" in human


def test_llm_reviewer_accepts_dict_output_and_rejects_malformed_output():
    ok = LLMReportReviewer(FakeStructuredLLM({"decision": "revise", "feedback": "x"}))
    assert ok.review("q", "r", [A]).decision == ReviewDecision.REVISE
    with pytest.raises(ValueError):
        LLMReportReviewer(FakeStructuredLLM({"decision": "maybe"})).review("q", "r", [A])


def test_normalize_gives_revisions_default_feedback_and_bounds_length():
    assert normalize_verdict(revise("   ")).feedback != ""
    assert len(normalize_verdict(revise("x" * 10_000)).feedback) == 2000
    assert normalize_verdict(APPROVE).feedback == ""


def test_node_reviews_the_draft_against_usable_sources_only():
    reviewer = ScriptedReviewer([APPROVE])
    state = new_state("q")
    state.update(sources=[A, THIN], draft=GOOD_DRAFT)

    update = make_reviewer_node(reviewer, 2)(state)

    assert update["review"] == APPROVE
    assert reviewer.calls[0]["sources"] == [A]
    assert reviewer.calls[0]["report"] == GOOD_DRAFT
    assert "errors" not in update


def test_node_flags_unresolved_concerns_when_budget_is_spent():
    state = new_state("q")
    state.update(sources=[A], draft=GOOD_DRAFT, revision_count=2)

    update = make_reviewer_node(ScriptedReviewer([revise("claim 2 is unsupported")]), 2)(state)

    assert update["review"].decision == ReviewDecision.REVISE
    assert "claim 2 is unsupported" in update["errors"][0]


def test_reviewer_failure_keeps_the_report_but_says_it_was_not_reviewed():
    state = new_state("q")
    state.update(sources=[A], draft=GOOD_DRAFT)

    update = make_reviewer_node(ScriptedReviewer([RuntimeError("overloaded")]), 2)(state)

    assert update["review"] is None
    assert update["status"] == ResearchStatus.REVIEWING
    assert "not reviewed" in update["errors"][0] and "overloaded" in update["errors"][0]


# --- full loop through the graph ---------------------------------------------------------------


def test_approved_draft_finishes_without_a_revision():
    writer, reviewer = ScriptedWriter([GOOD_DRAFT]), ScriptedReviewer([APPROVE])
    final = run(writer, reviewer)

    assert final["review"] == APPROVE
    assert final["report"].startswith("Alpha is described")
    assert final["revision_count"] == 0
    assert len(writer.calls) == 1 and len(reviewer.calls) == 1
    assert final["errors"] == []


def test_reviewer_feedback_reaches_the_writer_and_the_second_draft_is_approved():
    second = f"Alpha is described more carefully by the only source we have [{A.id}]."
    writer = ScriptedWriter([GOOD_DRAFT, second])
    reviewer = ScriptedReviewer([revise("be more careful"), APPROVE])

    final = run(writer, reviewer)

    assert writer.calls[1]["feedback"] == "be more careful"
    assert writer.calls[1]["previous_draft"] == GOOD_DRAFT
    assert reviewer.calls[1]["report"] == second
    assert final["revision_count"] == 1
    assert final["review"] == APPROVE
    assert final["report"].startswith("Alpha is described more carefully")


def test_reviewer_that_never_approves_stops_at_the_revision_budget_and_keeps_the_report():
    writer = ScriptedWriter([GOOD_DRAFT])
    reviewer = ScriptedReviewer([revise("still not good enough")])

    final = run(writer, reviewer, max_revisions=2)

    assert len(writer.calls) == 3 and len(reviewer.calls) == 3  # first draft + 2 revisions
    assert final["revision_count"] == 2
    assert final["status"] == ResearchStatus.REVIEWING
    assert final["report"] != ""
    assert final["review"].decision == ReviewDecision.REVISE
    assert any("reviewer concerns remain after 2 revision" in e for e in final["errors"])


def test_zero_revisions_allows_a_single_review():
    writer, reviewer = ScriptedWriter([GOOD_DRAFT]), ScriptedReviewer([revise("nope")])
    final = run(writer, reviewer, max_revisions=0)
    assert len(writer.calls) == 1 and len(reviewer.calls) == 1
    assert final["report"] != ""


def test_grounding_and_reviewer_share_one_revision_budget():
    ungrounded = "A claim with a fabricated citation attached to it [deadbeef]."
    writer = ScriptedWriter([ungrounded, GOOD_DRAFT])
    reviewer = ScriptedReviewer([revise("tighten it")])

    final = run(writer, reviewer, max_revisions=2)

    # draft 1 fails grounding (rev 1), draft 2 reviewed -> revise (rev 2), draft 3 is the
    # same grounded text, reviewed again, and the budget is gone.
    assert len(writer.calls) == 3
    assert final["revision_count"] == 2
    assert len(reviewer.calls) == 2


def test_a_rewrite_that_fails_grounding_after_review_discards_the_earlier_report():
    ungrounded = "Another fabricated claim with a fake citation [deadbeef] inside."
    writer = ScriptedWriter([GOOD_DRAFT, ungrounded])
    reviewer = ScriptedReviewer([revise("rewrite it")])

    final = run(writer, reviewer, max_revisions=1)

    assert final["status"] == ResearchStatus.FAILED
    assert final["report"] == ""
