"""Shared LangGraph state passed between the supervisor and worker agents."""

import operator
from typing import Annotated, TypedDict

from research_assistant.models import ResearchStatus, ReviewVerdict, Source


def merge_sources(left: list[Source], right: list[Source]) -> list[Source]:
    """Reducer: union by source id, keeping first-seen order.

    If a source is seen twice, the copy that has extracted page content wins, so a
    later extraction step can enrich a source that search returned without content.
    """
    merged: dict[str, Source] = {}
    for source in [*left, *right]:
        existing = merged.get(source.id)
        if existing is None or (source.has_content and not existing.has_content):
            merged[source.id] = source
    return list(merged.values())


class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    sources: Annotated[list[Source], merge_sources]
    draft: str
    review: ReviewVerdict | None
    revision_count: int
    status: ResearchStatus
    next_agent: str
    errors: Annotated[list[str], operator.add]


def new_state(question: str) -> ResearchState:
    """Build a valid initial state for a research question."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    return ResearchState(
        question=question,
        sub_questions=[],
        sources=[],
        draft="",
        review=None,
        revision_count=0,
        status=ResearchStatus.PLANNING,
        errors=[],
    )
