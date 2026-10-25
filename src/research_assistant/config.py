"""Typed, environment-driven configuration.

All settings can be overridden with ``RESEARCH_``-prefixed environment variables
or a local ``.env`` file, e.g. ``RESEARCH_MAX_REVISIONS=3``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Workflow bounds. These keep the writer/reviewer loop and retrieval finite.
    max_revisions: int = Field(default=2, ge=0, le=10)
    max_sources: int = Field(default=8, ge=1, le=50)
    max_sub_questions: int = Field(default=4, ge=1, le=10)

    # Retrieval
    request_timeout_s: float = Field(default=15.0, gt=0, le=120)
    max_page_chars: int = Field(default=8000, ge=500)

    # Human-in-the-loop review gate
    require_human_review: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
