import pytest

from research_assistant.retrieval import DuckDuckGoSearch, SearchError


def _backend(items):
    return lambda query, max_results: items


def test_maps_results_to_sources():
    search = DuckDuckGoSearch(
        _backend([{"title": " LangGraph ", "href": "https://Example.com/a/", "body": "intro"}])
    )
    [source] = search.search("langgraph", 5)
    assert source.url == "https://example.com/a"
    assert source.title == "LangGraph"
    assert source.snippet == "intro"


def test_skips_invalid_urls_and_dedupes():
    search = DuckDuckGoSearch(
        _backend(
            [
                {"title": "x", "href": "javascript:alert(1)", "body": ""},
                {"title": "y", "href": "", "body": ""},
                {"title": "a", "href": "https://a.com/p", "body": ""},
                {"title": "a again", "href": "https://a.com/p/", "body": ""},
            ]
        )
    )
    assert [s.url for s in search.search("q", 5)] == ["https://a.com/p"]


def test_respects_max_results():
    items = [{"title": str(i), "href": f"https://a.com/{i}", "body": ""} for i in range(10)]
    assert len(DuckDuckGoSearch(_backend(items)).search("q", 3)) == 3


def test_backend_failure_becomes_search_error():
    def boom(query, max_results):
        raise RuntimeError("rate limited")

    with pytest.raises(SearchError, match="rate limited"):
        DuckDuckGoSearch(boom).search("q", 3)


def test_no_results_is_an_empty_list_not_an_error():
    def nothing(query, max_results):
        raise RuntimeError("No results found.")

    assert DuckDuckGoSearch(nothing).search("very obscure query", 3) == []


def test_blank_query_rejected():
    with pytest.raises(ValueError):
        DuckDuckGoSearch(_backend([])).search("  ", 3)
