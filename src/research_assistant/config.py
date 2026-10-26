"""Typed, environment-driven configuration.

All settings can be overridden with ``RESEARCH_``-prefixed environment variables
or a local ``.env`` file, e.g. ``RESEARCH_MAX_REVISIONS=3``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Workflow bounds. These keep the writer/reviewer loop and retrieval finite.
    max_revisions: int = Field(default=2, ge=0, le=10)
    max_sources: int = Field(default=8, ge=1, le=50)
    max_sub_questions: int = Field(default=4, ge=1, le=10)

    # Retrieval
    request_timeout_s: float = Field(default=15.0, gt=0, le=120)
    max_page_chars: int = Field(default=8000, ge=500)
    min_page_chars: int = Field(default=300, ge=0)

    # LLM (Anthropic). The key is read from ANTHROPIC_API_KEY or .env, never logged.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"
    max_output_tokens: int = Field(default=2048, ge=256, le=16000)
    llm_timeout_s: float = Field(default=60.0, gt=0, le=300)

    # Human-in-the-loop review gate
    require_human_review: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
