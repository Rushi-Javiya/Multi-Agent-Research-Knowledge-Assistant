"""Domain models shared by every agent node."""

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field


def normalize_url(url: str) -> str:
    """Canonicalise a URL so the same page always maps to the same source id."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def source_id_for(url: str) -> str:
    """Stable, short identifier used as the citation key for a source."""
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:8]


class Source(BaseModel):
    """A retrieved web page. The citation key is ``id``."""

    model_config = ConfigDict(frozen=True)

    id: str
    url: str
    title: str = ""
    snippet: str = ""
    content: str = ""
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_url(cls, url: str, **fields: str) -> "Source":
        return cls(id=source_id_for(url), url=normalize_url(url), **fields)

    @property
    def has_content(self) -> bool:
        return bool(self.content.strip())


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"


class ReviewVerdict(BaseModel):
    """Structured output of the reviewer agent."""

    decision: ReviewDecision
    feedback: str = ""


class GroundingReport(BaseModel):
    """Result of the deterministic citation check on a draft."""

    ok: bool
    cited_ids: list[str] = Field(default_factory=list)
    invalid_ids: list[str] = Field(default_factory=list)
    uncited_blocks: list[str] = Field(default_factory=list)
    coverage: float = 0.0
    problems: list[str] = Field(default_factory=list)


class ResearchStatus(StrEnum):
    PLANNING = "planning"
    RESEARCHING = "researching"
    WRITING = "writing"
    REVIEWING = "reviewing"
    AWAITING_HUMAN = "awaiting_human"
    DONE = "done"
    FAILED = "failed"
