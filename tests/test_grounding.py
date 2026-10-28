from research_assistant.agents.grounding_node import make_grounding_node
from research_assistant.agents.writer import make_writer_node
from research_assistant.graph import build_graph
from research_assistant.grounding import (
    extract_citations,
    feedback_for_writer,
    render_report,
    split_blocks,
    validate_grounding,
)
from research_assistant.models import ResearchStatus, ReviewDecision, Source
from research_assistant.state import new_state

A = Source.from_url("https://a.com/a", title="Source A", content="alpha facts")
B = Source.from_url("https://b.com/b", title="Source B", content="beta facts")
THIN = Source.from_url("https://c.com/c", snippet="snippet only")
SOURCES = [A, B, THIN]


def good_draft():
    return (
        "## Summary\n"
        f"Alpha is described in detail by the first source [{A.id}].\n\n"
        "## Findings\n"
        f"- Beta works differently according to the second source [{B.id}].\n"
        f"- Both sources agree on the overall picture [{A.id}][{B.id}].\n"
    )


def test_extract_citations_dedupes_in_order_and_ignores_other_brackets():
    text = f"x [{B.id}] y [{A.id}] z [{B.id}] [1] [UPPERCASE] [not-an-id]"
    assert extract_citations(text) == [B.id, A.id]


def test_split_blocks_skips_headings_and_short_fragments():
    text = (
        "## Heading\nA paragraph that is long\nenough to count.\n\n"
        "- bullet one is also long enough\n- tiny\n---"
    )
    assert split_blocks(text) == [
        "A paragraph that is long enough to count.",
        "- bullet one is also long enough",
    ]


def test_valid_report_passes():
    report = validate_grounding(good_draft(), SOURCES)
    assert report.ok and report.problems == []
    assert report.cited_ids == [A.id, B.id]
    assert report.coverage == 1.0


def test_report_without_citations_is_rejected():
    report = validate_grounding("## Summary\nThis claim has no citation at all, sadly.", SOURCES)
    assert not report.ok
    assert "no citations" in report.problems[0]


def test_invented_citation_is_rejected():
    draft = good_draft() + "\nA made up claim with a fake id attached [deadbeef].\n"
    report = validate_grounding(draft, SOURCES)
    assert not report.ok
    assert report.invalid_ids == ["deadbeef"]


def test_citing_a_snippet_only_source_is_invalid_because_the_writer_never_saw_it():
    draft = f"A claim supported only by a source with no content [{THIN.id}]."
    report = validate_grounding(draft, SOURCES)
    assert not report.ok
    assert report.invalid_ids == [THIN.id]


def test_low_citation_coverage_is_rejected_and_uncited_blocks_reported():
    draft = (
        f"First cited statement made here [{A.id}].\n\n"
        "Second statement that has no citation whatsoever.\n\n"
        "Third statement that also has no citation here."
    )
    report = validate_grounding(draft, SOURCES, min_coverage=0.8)
    assert not report.ok
    assert report.coverage == 0.333
    assert len(report.uncited_blocks) == 2


def test_coverage_threshold_is_configurable():
    draft = (
        f"First cited statement made here [{A.id}].\n\n"
        "Second statement that has no citation whatsoever."
    )
    assert not validate_grounding(draft, SOURCES, min_coverage=0.8).ok
    assert validate_grounding(draft, SOURCES, min_coverage=0.5).ok


def test_empty_report_is_rejected():
    assert not validate_grounding("", SOURCES).ok
    assert not validate_grounding("## Summary\n", SOURCES).ok


def test_feedback_names_problems_and_allowed_ids_but_not_snippet_only_ones():
    report = validate_grounding("Uncited claim number one right here.", SOURCES)
    feedback = feedback_for_writer(report, SOURCES)
    assert "no citations" in feedback
    assert "Uncited claim number one" in feedback
    assert A.id in feedback and B.id in feedback and THIN.id not in feedback


def test_render_report_lists_only_cited_sources_in_citation_order():
    draft = f"Only the second source matters here [{B.id}]."
    rendered = render_report(draft, SOURCES, [B.id])
    assert rendered.endswith(f"## Sources\n- [{B.id}] Source B - {B.url}\n")
    assert A.url not in rendered


# --- graph integration: the gate, the bounded retry, and termination ---------------------------


class ScriptedWriter:
    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.calls = []

    def write(self, question, sources, feedback=None, previous_draft=""):
        self.calls.append({"feedback": feedback, "previous_draft": previous_draft})
        return self.drafts[min(len(self.calls), len(self.drafts)) - 1]


def researcher_stub(state):
    return {"sources": [A, B, THIN], "status": ResearchStatus.WRITING}


def run(writer, max_revisions=2):
    graph = build_graph(
        researcher_stub,
        make_writer_node(writer),
        make_grounding_node(max_revisions=max_revisions, min_coverage=0.8),
    )
    return graph.invoke(new_state("q"))


def test_grounded_first_draft_ends_with_a_report_and_sources_section():
    writer = ScriptedWriter([good_draft()])
    final = run(writer)

    assert final["status"] == ResearchStatus.REVIEWING
    assert final["report"].startswith("## Summary")
    assert "## Sources" in final["report"]
    assert final["grounding"].ok
    assert final["revision_count"] == 0
    assert len(writer.calls) == 1


def test_ungrounded_draft_triggers_a_revision_with_concrete_feedback_then_passes():
    bad = "A claim with a fabricated citation attached to it [deadbeef]."
    writer = ScriptedWriter([bad, good_draft()])
    final = run(writer)

    assert len(writer.calls) == 2
    assert "deadbeef" in writer.calls[1]["feedback"]
    assert writer.calls[1]["previous_draft"] == bad
    assert final["revision_count"] == 1
    assert final["status"] == ResearchStatus.REVIEWING
    assert final["review"] is None
    assert "deadbeef" not in final["report"]


def test_never_grounded_writer_terminates_after_the_revision_budget_and_fails():
    writer = ScriptedWriter(["Always uncited, never grounded, never valid at all."])
    final = run(writer, max_revisions=2)

    assert len(writer.calls) == 3  # first draft + 2 revisions, then it stops
    assert final["status"] == ResearchStatus.FAILED
    assert final["report"] == ""
    assert any("no grounded report after 2 revision" in e for e in final["errors"])
    assert not final["grounding"].ok


def test_zero_revisions_means_a_single_attempt():
    writer = ScriptedWriter(["Uncited text that should be rejected immediately."])
    final = run(writer, max_revisions=0)
    assert len(writer.calls) == 1
    assert final["status"] == ResearchStatus.FAILED


def test_node_requests_revision_while_budget_remains():
    state = new_state("q")
    state.update(sources=[A, B], draft="Uncited claim without any citation present.")
    update = make_grounding_node(2, 0.8)(state)
    assert update["review"].decision == ReviewDecision.REVISE
    assert "status" not in update
