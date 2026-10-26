"""Command line entry point."""

import argparse
import sys
from collections.abc import Sequence

from research_assistant.config import get_settings
from research_assistant.logging_config import configure_logging
from research_assistant.retrieval import DuckDuckGoSearch
from research_assistant.retrieval.fetch import PageFetcher
from research_assistant.retrieval.pipeline import retrieve


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    cmd = sub.add_parser("retrieve", help="search the web and extract page content for a question")
    cmd.add_argument("question")
    cmd.add_argument("--max-sources", type=int, help="override RESEARCH_MAX_SOURCES")
    cmd.add_argument("--preview-chars", type=int, default=300, help="characters of content to show")
    return parser


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
        )
    finally:
        fetcher.close()

    for i, source in enumerate(result.sources, 1):
        print(f"\n[{i}] {source.title or '(untitled)'}  <{source.id}>")
        print(f"    {source.url}")
        body = source.content or source.snippet
        label = f"{len(source.content)} chars" if source.has_content else "snippet only"
        print(f"    ({label}) {body[:preview_chars].replace(chr(10), ' ')}")

    if result.errors:
        print(f"\n{len(result.errors)} problem(s):", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
    return 0 if result.sources else 1


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    configure_logging(get_settings().log_level)
    if args.command == "retrieve":
        return _run_retrieve(args.question, args.max_sources, args.preview_chars)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
