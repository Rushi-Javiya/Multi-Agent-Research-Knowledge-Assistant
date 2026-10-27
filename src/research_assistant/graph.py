"""LangGraph workflow assembly. Later milestones add the reviewer and a supervisor."""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_assistant.agents.common import NodeFn
from research_assistant.models import ResearchStatus
from research_assistant.state import ResearchState


def route_after_research(state: ResearchState) -> Literal["writer", "end"]:
    """Don't write a report when research produced nothing usable."""
    return "end" if state.get("status") == ResearchStatus.FAILED else "writer"


def build_graph(
    researcher: NodeFn, writer: NodeFn
) -> CompiledStateGraph[ResearchState, Any, Any, Any]:
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher)  # type: ignore[call-overload]
    graph.add_node("writer", writer)  # type: ignore[call-overload]
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges(
        "researcher", route_after_research, {"writer": "writer", "end": END}
    )
    graph.add_edge("writer", END)
    return graph.compile()
