"""Search, fetch and extract in one step, preserving every source."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from research_assistant.logging_config import get_logger
from research_assistant.models import Source
from research_assistant.retrieval.base import SearchError, SearchProvider
from research_assistant.retrieval.extract import extract_text
from research_assistant.retrieval.fetch import FetchError, PageFetcher

log = get_logger("retrieval")


@dataclass
class RetrievalResult:
    sources: list[Source] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _enrich(
    source: Source, fetcher: PageFetcher, max_chars: int, min_chars: int
) -> tuple[Source, str | None]:
    try:
        title, text = extract_text(fetcher.fetch(source.url), max_chars)
    except FetchError as exc:
        return source, f"{source.url}: {exc}"
    if not text:
        return source, f"{source.url}: no extractable text"
    if len(text) < min_chars:
        return source, f"{source.url}: too little text ({len(text)} chars)"
    return source.model_copy(update={"title": title or source.title, "content": text}), None


def retrieve(
    query: str,
    search: SearchProvider,
    fetcher: PageFetcher,
    max_sources: int,
    max_page_chars: int,
    min_page_chars: int = 0,
    workers: int = 4,
) -> RetrievalResult:
    """Never raises for provider/page failures: they are reported in ``errors``.

    A source whose page can't be fetched is kept (with its search snippet) rather than
    dropped, so the failure is visible and nothing is silently lost.
    """
    try:
        found = search.search(query, max_sources)
    except SearchError as exc:
        log.warning("search failed: %s", exc)
        return RetrievalResult(errors=[str(exc)])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(lambda s: _enrich(s, fetcher, max_page_chars, min_page_chars), found)
        )

    errors = [err for _, err in results if err]
    for err in errors:
        log.warning("fetch problem: %s", err)
    return RetrievalResult(sources=[source for source, _ in results], errors=errors)
