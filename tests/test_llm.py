import pytest
from langchain_anthropic import ChatAnthropic

from research_assistant.config import Settings
from research_assistant.llm import LLMConfigError, build_llm


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        build_llm(Settings(_env_file=None))


def test_builds_anthropic_model_from_settings():
    settings = Settings(_env_file=None, anthropic_api_key="sk-test", llm_model="some-model")
    llm = build_llm(settings)
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "some-model"


def test_api_key_is_not_exposed_in_repr():
    settings = Settings(_env_file=None, anthropic_api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(settings)
