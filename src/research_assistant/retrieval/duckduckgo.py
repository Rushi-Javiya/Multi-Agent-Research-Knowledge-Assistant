"""DuckDuckGo search provider (no API key required)."""

from collections.abc import Callable
from typing import Any

from research_assistant.models import Source
from research_assistant.retrieval.base import SearchError


def _default_backend(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    return list(DDGS().text(query, max_results=max_results))


class DuckDuckGoSearch:
    def __init__(
        self, backend: Callable[[str, int], list[dict[str, Any]]] = _default_backend
    ) -> None:
        self._backend = backend

    def search(self, query: str, max_results: int) -> list[Source]:
        if not query.strip():
            raise ValueError("query must not be empty")
        try:
            raw = self._backend(query, max_results)
        except Exception as exc:
            if "no results" in str(exc).lower():
                return []
            raise SearchError(f"DuckDuckGo search failed: {exc}") from exc

        sources: dict[str, Source] = {}
        for item in raw:
            url = (item.get("href") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            source = Source.from_url(
                url,
                title=(item.get("title") or "").strip(),
                snippet=(item.get("body") or "").strip(),
            )
            sources.setdefault(source.id, source)
        return list(sources.values())[:max_results]
