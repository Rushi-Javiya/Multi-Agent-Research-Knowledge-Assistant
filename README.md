# Multi-Agent Research & Knowledge Assistant

A stateful multi-agent research workflow built with **LangGraph**. A supervisor routes a
question through researcher, writer and reviewer agents that share one execution state, and
produces a **citation-based report in which every claim is grounded in a retrieved source**.

> Status: under active development. See the roadmap below for completed and upcoming targets.

## Goals

- **Grounded by construction** - a deterministic validator rejects any report whose citations
  don't resolve to a retrieved source. It does not rely on the model to police itself.
- **Bounded** - the writer/reviewer refinement loop and retrieval have hard limits, so a run always terminates.
- **Human in the loop** - reports pause for approval using LangGraph interrupts and resume commands.
- **Testable** - every workflow property (grounding, termination, review, revision) has automated tests.

## Architecture (target)

```
            +-------------+
 question ->| supervisor  |<----------------------+
            +------+------+                       |
        +----------+-----------+                  |
        v                      v                  |
  +------------+        +------------+       +----------+
  | researcher |------->|   writer   |<----->| reviewer |
  +------------+        +------------+       +----------+
   search + extract       cited draft          grounding check
                                               + human interrupt
```

Shared state (`src/research_assistant/state.py`) carries the question, sub-questions, sources
(merged and deduplicated by a reducer), draft, review verdict, revision count and errors.

## Roadmap

- [x] Project scaffold, typed config, domain models, shared graph state, CI
- [x] Search provider abstraction and structured source retrieval
- [x] Page content extraction, cleaning and source preservation
- [x] Researcher agent: sub-question planning, retrieval, error handling
- [x] Writer agent with citation-based report generation
- [x] Deterministic grounding validator that rejects invalid citations
- [ ] Reviewer agent and bounded writer-reviewer refinement loop
- [ ] Supervisor with conditional routing and full graph assembly
- [ ] Human-in-the-loop review via `interrupt` and `Command(resume=...)`
- [ ] Docker image, integration tests and final documentation

## Getting started

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e ".[dev]"
cp .env.example .env
pytest
ruff check .
```

## Try the retrieval layer

No API key is needed for retrieval (it uses DuckDuckGo):

```bash
research-assistant retrieve "What is LangGraph and how does it work?" --max-sources 5
```

It searches, fetches each page, extracts the main text and prints every source with its
citation id. Pages that can't be fetched (403, timeouts, non-HTML) are kept as snippet-only
sources and reported, never silently dropped.

**Security:** URLs come from search results, so the fetcher refuses non-http(s) schemes and any host
that resolves to a private, loopback or link-local address, checking every redirect hop. Response
size and content type are limited.

Extraction prefers `<article>`/`<main>` content. For pages without them it drops short
fragments (menu items), but longer menu text can still slip through. Pages with too little
text (see `RESEARCH_MIN_PAGE_CHARS`) are demoted to snippet-only so they can't be cited as evidence.

## Try the researcher and writer agents (needs an Anthropic key)

Put your key in `.env` (never commit it):

```
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
research-assistant research "How does LangGraph human-in-the-loop work?"
```

Claude splits the question into focused search queries, each is retrieved, and sources are
merged and deduplicated. The run is marked `failed` if no source has usable content, because a
grounded report would be impossible. If planning fails the researcher falls back to the original
question rather than crashing.

The writer then drafts a Markdown report from the sources that have extracted content,
citing each claim by source id, e.g. `[7dd57631]`. Page text is passed to the model inside
delimited `<source>` blocks and treated as untrusted data, so instructions hidden in a web page
are not followed.

### Grounding gate

Every draft passes a **deterministic** check (`grounding.py`, no LLM involved) before it can
become the report. A draft is rejected if:

- it has no citations,
- it cites an id that is not one of the sources the writer was given (invented ids, or sources
  that had no extracted content), or
- fewer than `RESEARCH_MIN_CITATION_COVERAGE` (default 80%) of its paragraphs and bullets are cited.

A rejected draft goes back to the writer with concrete feedback (the bad ids, the uncited
paragraphs, the allowed ids). This repeats at most `RESEARCH_MAX_REVISIONS` times (default 2); after
that the run ends as `failed` rather than returning an ungrounded report. An accepted report gets a
`## Sources` section generated by code from the sources it actually cites.

**What this does not prove:** the check guarantees that citations are real and present, not that a
cited source truly supports the sentence. Judging that is the reviewer agent's job (next target).

## Configuration

Settings are read from `RESEARCH_*` environment variables or `.env` (see `.env.example`).
Bounds such as `RESEARCH_MAX_REVISIONS` are validated at startup.

## Project layout

```
src/research_assistant/
  config.py          typed settings
  models.py          Source, ReviewVerdict, ResearchStatus
  state.py           LangGraph shared state + reducers
  logging_config.py  logging setup
  cli.py             command line entry point
  llm.py             Anthropic chat model factory
  graph.py           LangGraph workflow assembly and routing
  grounding.py       deterministic citation validator
  agents/            planner and researcher agent nodes
  retrieval/         search providers, safe page fetcher, text extraction, pipeline
tests/
```

## License

MIT
