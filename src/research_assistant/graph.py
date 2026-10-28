"""LangGraph workflow assembly. Later milestones add the reviewer and a supervisor."""

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


def route_after_grounding(state: ResearchState) -> Literal["writer", "end"]:
    """Loop back to the writer only when the check asked for a revision.

    The grounding node marks the run failed once the revision budget is spent, so the loop
    always terminates.
    """
    review = state.get("review")
    wants_revision = review is not None and review.decision == ReviewDecision.REVISE
    return "writer" if wants_revision and not _failed(state) else "end"


def build_graph(
    researcher: NodeFn, writer: NodeFn, grounding: NodeFn
) -> CompiledStateGraph[ResearchState, Any, Any, Any]:
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher)  # type: ignore[call-overload]
    graph.add_node("writer", writer)  # type: ignore[call-overload]
    graph.add_node("grounding", grounding)  # type: ignore[call-overload]
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges(
        "researcher", route_after_research, {"writer": "writer", "end": END}
    )
    graph.add_conditional_edges(
        "writer", route_after_writer, {"grounding": "grounding", "end": END}
    )
    graph.add_conditional_edges(
        "grounding", route_after_grounding, {"writer": "writer", "end": END}
    )
    return graph.compile()
