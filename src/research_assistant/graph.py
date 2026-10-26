"""LangGraph workflow assembly. Later milestones add writer, reviewer and routing."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_assistant.agents.researcher import NodeFn
from research_assistant.state import ResearchState


def build_graph(researcher: NodeFn) -> CompiledStateGraph[ResearchState, Any, Any, Any]:
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher)  # type: ignore[call-overload]
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", END)
    return graph.compile()
