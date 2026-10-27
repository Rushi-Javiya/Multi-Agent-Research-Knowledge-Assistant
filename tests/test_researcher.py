import httpx

from research_assistant.agents.researcher import make_researcher_node
from research_assistant.agents.writer import make_writer_node
from research_assistant.config import Settings
from research_assistant.graph import build_graph
from research_assistant.models import ResearchStatus, Source
from research_assistant.retrieval.fetch import PageFetcher
from research_assistant.state import new_state


class FakePlanner:
    def __init__(self, questions=None, error=None):
        self.questions = questions or []
        self.error = error

    def plan(self, question, max_questions):
        if self.error:
            raise self.error
        return self.questions[:max_questions]


class FakeSearch:
    """Maps each query to a list of URLs on a.com."""

    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, max_results):
        self.queries.append(query)
        return [Source.from_url(f"https://a.com/{p}") for p in self.results.get(query, [])]


def handler(request):
    if request.url.path.startswith("/good"):
        body = f"<article><p>Content of {request.url.path}</p></article>"
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})
    return httpx.Response(404)


def make_fetcher():
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return PageFetcher(client=client, resolver=lambda h, p: [(2, 1, 6, "", ("93.184.216.34", 0))])


def settings(**overrides):
    base = {"max_sources": 6, "min_page_chars": 0, "max_sub_questions": 3}
    return Settings(_env_file=None, **{**base, **overrides})


class StubWriter:
    def write(self, question, sources, feedback=None, previous_draft=""):
        return "report"


def run(planner, search, **overrides):
    node = make_researcher_node(planner, search, make_fetcher(), settings(**overrides))
    return build_graph(node, make_writer_node(StubWriter())).invoke(new_state("original question"))


def test_plans_retrieves_and_merges_sources_across_sub_questions():
    search = FakeSearch({"q1": ["good1", "good2"], "q2": ["good2", "good3"]})
    final = run(FakePlanner(["q1", "q2"]), search)

    assert final["sub_questions"] == ["q1", "q2"]
    assert search.queries == ["q1", "q2"]
    assert [s.url for s in final["sources"]] == [
        "https://a.com/good1", "https://a.com/good2", "https://a.com/good3",
    ]
    assert final["status"] == ResearchStatus.REVIEWING
    assert final["errors"] == []


def test_planner_failure_falls_back_to_original_question():
    search = FakeSearch({"original question": ["good1"]})
    final = run(FakePlanner(error=RuntimeError("llm down")), search)

    assert final["sub_questions"] == ["original question"]
    assert final["status"] == ResearchStatus.REVIEWING
    assert any("planning failed" in e and "llm down" in e for e in final["errors"])


def test_empty_plan_falls_back_to_original_question():
    final = run(FakePlanner([]), FakeSearch({"original question": ["good1"]}))
    assert final["sub_questions"] == ["original question"]


def test_no_usable_sources_marks_run_failed():
    final = run(FakePlanner(["q1"]), FakeSearch({"q1": ["bad1", "bad2"]}))

    assert final["status"] == ResearchStatus.FAILED
    assert len(final["sources"]) == 2
    assert any("no usable sources" in e for e in final["errors"])


def test_failed_research_skips_the_writer():
    class ExplodingWriter:
        def write(self, *args, **kwargs):
            raise AssertionError("writer must not run when research failed")

    node = make_researcher_node(
        FakePlanner(["q1"]), FakeSearch({"q1": ["bad1"]}), make_fetcher(), settings()
    )
    final = build_graph(node, make_writer_node(ExplodingWriter())).invoke(new_state("q"))
    assert final["status"] == ResearchStatus.FAILED
    assert final["draft"] == ""


def test_usable_sources_rank_first_and_total_is_bounded():
    search = FakeSearch({"q1": ["bad1", "bad2", "good1", "good2"]})
    final = run(FakePlanner(["q1"]), search, max_sources=3)

    assert len(final["sources"]) == 3
    assert [s.has_content for s in final["sources"]] == [True, True, False]
