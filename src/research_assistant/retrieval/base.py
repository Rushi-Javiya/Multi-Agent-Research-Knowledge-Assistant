"""Interfaces for the retrieval layer."""

from typing import Protocol

from research_assistant.models import Source


class SearchError(Exception):
    """Raised when a search provider fails. Callers decide how to degrade."""


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[Source]:
        """Return structured sources (url, title, snippet) for a query."""
        ...
