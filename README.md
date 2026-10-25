# Multi-Agent Research & Knowledge Assistant

A stateful multi-agent research workflow built with **LangGraph**. A supervisor routes a
question through researcher, writer and reviewer agents that share one execution state, and
produces a **citation-based report in which every claim is grounded in a retrieved source**.

> Status: under active development, with one tested milestone per day. See the roadmap below.

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

| Day | Milestone | Status |
|----:|-----------|:------:|
| 1 | Project scaffold, typed config, domain models, shared graph state, CI | Done |
| 2 | Search provider abstraction + structured source retrieval | |
| 3 | Page content extraction, cleaning, source preservation | |
| 4 | Researcher node (planning sub-questions, retrieval, error handling) | |
| 5 | Writer node with citation-based report generation | |
| 6 | Deterministic grounding validator | |
| 7 | Reviewer node + bounded writer-reviewer refinement loop | |
| 8 | Supervisor + conditional routing, full graph assembly | |
| 9 | Human-in-the-loop review via `interrupt` and `Command(resume=...)` | |
| 10 | CLI, Docker, integration tests, final docs | |

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
tests/
```

## License

MIT
