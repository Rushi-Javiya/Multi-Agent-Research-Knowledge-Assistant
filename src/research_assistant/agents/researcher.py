"""Researcher agent: plans sub-questions, then retrieves and extracts sources."""

import math
from collections.abc import Callable
from typing import Any

from research_assistant.agents.planner import QuestionPlanner
from research_assistant.config import Settings
from research_assistant.logging_config import get_logger
from research_assistant.models import ResearchStatus
from research_assistant.retrieval.base import SearchProvider
from research_assistant.retrieval.fetch import PageFetcher
from research_assistant.retrieval.pipeline import retrieve
from research_assistant.state import ResearchState, merge_sources

log = get_logger("researcher")

NodeFn = Callable[[ResearchState], dict[str, Any]]


def make_researcher_node(
    planner: QuestionPlanner,
    search: SearchProvider,
    fetcher: PageFetcher,
    settings: Settings,
) -> NodeFn:
    def researcher(state: ResearchState) -> dict[str, Any]:
        question = state["question"]
        errors: list[str] = []

        try:
            sub_questions = planner.plan(question, settings.max_sub_questions)
        except Exception as exc:  # planner talks to an LLM; any failure must degrade, not crash
            log.warning("planning failed: %s", exc)
            errors.append(f"planning failed: {exc}")
            sub_questions = []
        if not sub_questions:
            sub_questions = [question]

        per_query = max(2, math.ceil(settings.max_sources / len(sub_questions)))
        sources = state.get("sources", [])
        for sub_question in sub_questions:
            result = retrieve(
                sub_question,
                search,
                fetcher,
                max_sources=per_query,
                max_page_chars=settings.max_page_chars,
                min_page_chars=settings.min_page_chars,
            )
            sources = merge_sources(sources, result.sources)
            errors.extend(result.errors)

        # Keep usable sources first, then bound the total.
        sources = sorted(sources, key=lambda s: not s.has_content)[: settings.max_sources]

        update: dict[str, Any] = {"sub_questions": sub_questions, "sources": sources}
        if any(s.has_content for s in sources):
            update["status"] = ResearchStatus.WRITING
        else:
            update["status"] = ResearchStatus.FAILED
            errors.append("no usable sources were retrieved, so a grounded report is impossible")
        update["errors"] = errors
        return update

    return researcher
