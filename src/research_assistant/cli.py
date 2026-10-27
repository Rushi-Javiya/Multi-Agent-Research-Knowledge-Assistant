"""Command line entry point."""

import argparse
import sys
from collections.abc import Sequence

from research_assistant.agents.planner import LLMPlanner
from research_assistant.agents.researcher import make_researcher_node
from research_assistant.agents.writer import LLMReportWriter, make_writer_node
from research_assistant.config import get_settings
from research_assistant.graph import build_graph
from research_assistant.llm import LLMConfigError, build_llm
from research_assistant.logging_config import configure_logging
from research_assistant.models import Source
from research_assistant.retrieval import DuckDuckGoSearch
from research_assistant.retrieval.fetch import PageFetcher
from research_assistant.retrieval.pipeline import retrieve
from research_assistant.state import new_state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    cmd = sub.add_parser("retrieve", help="search the web and extract page content for a question")
    cmd.add_argument("question")
    cmd.add_argument("--max-sources", type=int, help="override RESEARCH_MAX_SOURCES")
    cmd.add_argument("--preview-chars", type=int, default=300, help="characters of content to show")

    cmd = sub.add_parser(
        "research", help="plan, retrieve sources, and write a cited report with Claude"
    )
    cmd.add_argument("question")
    cmd.add_argument("--preview-chars", type=int, default=200, help="characters of content to show")
    return parser


def _print_sources(sources: list[Source], preview_chars: int) -> None:
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}] {source.title or '(untitled)'}  <{source.id}>")
        print(f"    {source.url}")
        body = source.content or source.snippet
        label = f"{len(source.content)} chars" if source.has_content else "snippet only"
        print(f"    ({label}) {body[:preview_chars].replace(chr(10), ' ')}")


def _print_errors(errors: list[str]) -> None:
    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)


def _run_retrieve(question: str, max_sources: int | None, preview_chars: int) -> int:
    settings = get_settings()
    fetcher = PageFetcher(timeout=settings.request_timeout_s)
    try:
        result = retrieve(
            question,
            DuckDuckGoSearch(),
            fetcher,
            max_sources=max_sources or settings.max_sources,
            max_page_chars=settings.max_page_chars,
            min_page_chars=settings.min_page_chars,
        )
    finally:
        fetcher.close()

    _print_sources(result.sources, preview_chars)
    _print_errors(result.errors)
    return 0 if result.sources else 1


def _run_research(question: str, preview_chars: int) -> int:
    settings = get_settings()
    try:
        llm = build_llm(settings)
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    fetcher = PageFetcher(timeout=settings.request_timeout_s)
    try:
        researcher = make_researcher_node(LLMPlanner(llm), DuckDuckGoSearch(), fetcher, settings)
        writer = make_writer_node(LLMReportWriter(llm))
        final = build_graph(researcher, writer).invoke(new_state(question))
    finally:
        fetcher.close()

    print("Sub-questions:")
    for sub_question in final["sub_questions"]:
        print(f"  - {sub_question}")
    _print_sources(final["sources"], preview_chars)
    if final.get("draft"):
        print(f"\n{'=' * 70}\n{final['draft']}\n{'=' * 70}")
    _print_errors(final["errors"])
    print(f"\nstatus: {final['status']}")
    return 0 if final["status"] != "failed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    configure_logging(get_settings().log_level)
    if args.command == "retrieve":
        return _run_retrieve(args.question, args.max_sources, args.preview_chars)
    if args.command == "research":
        return _run_research(args.question, args.preview_chars)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
