import httpx

from research_assistant.models import Source
from research_assistant.retrieval import SearchError
from research_assistant.retrieval.fetch import PageFetcher
from research_assistant.retrieval.pipeline import retrieve


class FakeSearch:
    def __init__(self, sources=None, error=None):
        self._sources = sources or []
        self._error = error

    def search(self, query, max_results):
        if self._error:
            raise self._error
        return self._sources[:max_results]


def fetcher(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return PageFetcher(client=client, resolver=lambda h, p: [(2, 1, 6, "", ("93.184.216.34", 0))])


def handler(request):
    if request.url.path == "/ok":
        body = "<title>Good</title><article><p>Real content</p></article>"
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})
    if request.url.path == "/empty":
        return httpx.Response(200, text="<script>x</script>", headers={"content-type": "text/html"})
    return httpx.Response(500)


def test_enriches_sources_and_preserves_failures():
    found = [
        Source.from_url("https://a.com/ok", title="snippet title", snippet="s1"),
        Source.from_url("https://a.com/boom", snippet="s2"),
        Source.from_url("https://a.com/empty", snippet="s3"),
    ]
    result = retrieve("q", FakeSearch(found), fetcher(handler), max_sources=5, max_page_chars=500)

    good, failed, empty = result.sources
    assert (good.title, good.content) == ("Good", "Real content")
    assert not failed.has_content and failed.snippet == "s2"
    assert not empty.has_content
    assert len(result.errors) == 2
    assert [s.id for s in result.sources] == [s.id for s in found]


def test_search_failure_is_reported_not_raised():
    search = FakeSearch(error=SearchError("down"))
    result = retrieve("q", search, fetcher(handler), max_sources=5, max_page_chars=500)
    assert result.sources == []
    assert result.errors == ["down"]
