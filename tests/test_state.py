import pytest
from langgraph.graph import END, START, StateGraph

from research_assistant.models import ResearchStatus, Source
from research_assistant.state import ResearchState, merge_sources, new_state


def test_new_state_is_valid_and_trimmed():
    state = new_state("  What is RAG?  ")
    assert state["question"] == "What is RAG?"
    assert state["status"] == ResearchStatus.PLANNING
    assert state["revision_count"] == 0
    assert state["sources"] == []


def test_new_state_rejects_blank_question():
    with pytest.raises(ValueError):
        new_state("   ")


def test_merge_sources_dedupes_and_keeps_order():
    a = Source.from_url("https://a.com")
    b = Source.from_url("https://b.com")
    assert [s.url for s in merge_sources([a, b], [a])] == [a.url, b.url]


def test_merge_sources_prefers_copy_with_content():
    bare = Source.from_url("https://a.com")
    full = Source.from_url("https://a.com", content="page text")
    merged = merge_sources([bare], [full])
    assert len(merged) == 1
    assert merged[0].content == "page text"


def test_reducers_apply_inside_a_real_langgraph():
    """Two nodes write sources/errors; the graph must merge, not overwrite."""

    def node_a(state: ResearchState) -> dict:
        return {"sources": [Source.from_url("https://a.com")], "errors": ["a failed"]}

    def node_b(state: ResearchState) -> dict:
        return {
            "sources": [Source.from_url("https://a.com"), Source.from_url("https://b.com")],
            "errors": ["b failed"],
        }

    graph = StateGraph(ResearchState)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    result = graph.compile().invoke(new_state("q"))

    assert [s.url for s in result["sources"]] == ["https://a.com/", "https://b.com/"]
    assert result["errors"] == ["a failed", "b failed"]
