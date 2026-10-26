import pytest

from research_assistant.agents.planner import LLMPlanner, SubQuestionPlan, clean_sub_questions


class FakeStructuredLLM:
    """Stands in for ChatModel.with_structured_output(...) without any network call."""

    def __init__(self, result):
        self._result = result
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.kwargs = kwargs
        return self

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_clean_dedupes_trims_and_caps():
    items = ["  What  is RAG? ", "what is rag?", "", "   ", "How does it work", "Third", "Fourth"]
    assert clean_sub_questions(items, 3) == ["What is RAG?", "How does it work", "Third"]


def test_clean_bounds_query_length():
    [only] = clean_sub_questions(["x" * 1000], 5)
    assert len(only) == 200


def test_llm_planner_returns_cleaned_queries():
    llm = FakeStructuredLLM(SubQuestionPlan(sub_questions=["A ", "a", "B"]))
    assert LLMPlanner(llm).plan("question", 5) == ["A", "B"]
    assert llm.schema is SubQuestionPlan


def test_llm_planner_avoids_forced_tool_calling():
    llm = FakeStructuredLLM(SubQuestionPlan(sub_questions=["a"]))
    LLMPlanner(llm)
    assert llm.kwargs == {"method": "json_schema"}


def test_llm_planner_accepts_dict_output():
    llm = FakeStructuredLLM({"sub_questions": ["only one"]})
    assert LLMPlanner(llm).plan("q", 5) == ["only one"]


def test_llm_planner_tells_the_model_the_limit_and_passes_the_question():
    llm = FakeStructuredLLM(SubQuestionPlan(sub_questions=["a"]))
    LLMPlanner(llm).plan("my question", 3)
    (_, system), (_, human) = llm.messages
    assert "at most 3" in system
    assert human == "my question"


def test_llm_planner_rejects_malformed_output():
    with pytest.raises(ValueError):
        LLMPlanner(FakeStructuredLLM({"wrong": 1})).plan("q", 3)
