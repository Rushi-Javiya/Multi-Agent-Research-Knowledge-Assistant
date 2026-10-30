"""LangGraph workflow assembly. A later milestone adds a supervisor and human review.

researcher -> writer -> grounding -> reviewer, with a bounded loop back to the writer.
Grounding rejections and reviewer rejections spend the same revision budget, so the whole
loop is finite whichever of them keeps objecting.
"""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_assistant.agents.common import NodeFn
from research_assistant.models import ResearchStatus, ReviewDecision
from research_assistant.state import ResearchState


def _failed(state: ResearchState) -> bool:
    return state.get("status") == ResearchStatus.FAILED


def route_after_research(state: ResearchState) -> Literal["writer", "end"]:
    """Don't write a report when research produced nothing usable."""
    return "end" if _failed(state) else "writer"


def route_after_writer(state: ResearchState) -> Literal["grounding", "end"]:
    return "end" if _failed(state) else "grounding"


def route_after_grounding(state: ResearchState) -> Literal["reviewer", "writer", "end"]:
    """Grounded drafts go to the reviewer; rejected ones go back to the writer.

    The grounding node marks the run failed once the revision budget is spent, so a rejected
    draft only reaches the writer while budget remains.
    """
    if _failed(state):
        return "end"
    grounding = state.get("grounding")
    return "reviewer" if grounding is not None and grounding.ok else "writer"


def make_route_after_review(
    max_revisions: int,
) -> Callable[[ResearchState], Literal["writer", "end"]]:
    def route_after_review(state: ResearchState) -> Literal["writer", "end"]:
        review = state.get("review")
        wants_revision = review is not None and review.decision == ReviewDecision.REVISE
        has_budget = state.get("revision_count", 0) < max_revisions
        return "writer" if wants_revision and has_budget else "end"

    return route_after_review


def build_graph(
    researcher: NodeFn,
    writer: NodeFn,
    grounding: NodeFn,
    reviewer: NodeFn,
    max_revisions: int,
) -> CompiledStateGraph[ResearchState, Any, Any, Any]:
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher)  # type: ignore[call-overload]
    graph.add_node("writer", writer)  # type: ignore[call-overload]
    graph.add_node("grounding", grounding)  # type: ignore[call-overload]
    graph.add_node("reviewer", reviewer)  # type: ignore[call-overload]
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges(
        "researcher", route_after_research, {"writer": "writer", "end": END}
    )
    graph.add_conditional_edges(
        "writer", route_after_writer, {"grounding": "grounding", "end": END}
    )
    graph.add_conditional_edges(
        "grounding",
        route_after_grounding,
        {"reviewer": "reviewer", "writer": "writer", "end": END},
    )
    graph.add_conditional_edges(
        "reviewer", make_route_after_review(max_revisions), {"writer": "writer", "end": END}
    )
    return graph.compile()
