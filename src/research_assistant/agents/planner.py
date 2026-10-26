"""Break a research question into focused search queries."""

from typing import Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

_MAX_QUERY_CHARS = 200

_SYSTEM_PROMPT = (
    "You plan web research. Split the user's question into focused, self-contained search "
    "queries that together cover it. Each query must make sense on its own, avoid overlap, "
    "and be phrased the way you would type it into a search engine: plain keywords, "
    "at most 8 words, no quotes, no code syntax, no function or parameter names."
)


class QuestionPlanner(Protocol):
    def plan(self, question: str, max_questions: int) -> list[str]: ...


class SubQuestionPlan(BaseModel):
    sub_questions: list[str] = Field(description="Focused search queries, most important first.")


def clean_sub_questions(items: list[str], max_questions: int) -> list[str]:
    """Strip, drop blanks, dedupe case-insensitively, bound length and count."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        text = " ".join(item.split())[:_MAX_QUERY_CHARS]
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned[:max_questions]


class LLMPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        # json_schema avoids forced tool calling, which newer Claude models don't support.
        self._structured = llm.with_structured_output(SubQuestionPlan, method="json_schema")

    def plan(self, question: str, max_questions: int) -> list[str]:
        system = f"{_SYSTEM_PROMPT} Return at most {max_questions} queries."
        result = self._structured.invoke([("system", system), ("human", question)])
        if isinstance(result, SubQuestionPlan):
            plan = result
        else:
            plan = SubQuestionPlan.model_validate(result)
        return clean_sub_questions(plan.sub_questions, max_questions)
