"""Construction of the chat model used by the agents."""

from langchain_core.language_models import BaseChatModel

from research_assistant.config import Settings


class LLMConfigError(Exception):
    """The LLM cannot be built from the current configuration."""


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.anthropic_api_key is None:
        raise LLMConfigError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file (see .env.example)."
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_output_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )
