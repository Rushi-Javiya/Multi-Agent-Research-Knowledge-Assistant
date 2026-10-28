"""Deterministic grounding validation.

No model is involved: a report is accepted only if its citations resolve to sources the writer
was actually given, and enough of its content is cited. This is what stops invented ids or
uncited claims from reaching a reader.
"""

import re
from collections.abc import Sequence

from research_assistant.models import GroundingReport, Source

_CITATION = re.compile(r"\[([0-9a-f]{8})\]")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MIN_WORDS = 4
_PREVIEW_CHARS = 80


def extract_citations(text: str) -> list[str]:
    """Cited ids in order of first appearance, without duplicates."""
    return list(dict.fromkeys(_CITATION.findall(text)))


def split_blocks(text: str) -> list[str]:
    """Split Markdown into paragraphs and list items, skipping headings and fragments."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            current = None
        elif _LIST_ITEM.match(raw) or current is None:
            current = [line]
            blocks.append(current)
        else:
            current.append(line)
    joined = (" ".join(parts) for parts in blocks)
    return [b for b in joined if len(b.split()) >= _MIN_WORDS]


def validate_grounding(
    draft: str, sources: Sequence[Source], min_coverage: float = 0.8
) -> GroundingReport:
    # Only sources with extracted content were shown to the writer, so only they can be cited.
    allowed = {s.id for s in sources if s.has_content}
    cited = extract_citations(draft)
    invalid = [c for c in cited if c not in allowed]
    blocks = split_blocks(draft)
    uncited = [b for b in blocks if not _CITATION.search(b)]
    coverage = 1 - len(uncited) / len(blocks) if blocks else 0.0

    problems: list[str] = []
    if not cited:
        problems.append("the report contains no citations")
    if invalid:
        problems.append("cites ids that are not among the sources: " + ", ".join(invalid))
    if blocks and coverage < min_coverage:
        problems.append(
            f"only {coverage:.0%} of paragraphs and bullets are cited (need {min_coverage:.0%})"
        )
    if not blocks:
        problems.append("the report has no substantive content")

    return GroundingReport(
        ok=not problems,
        cited_ids=cited,
        invalid_ids=invalid,
        uncited_blocks=uncited,
        coverage=round(coverage, 3),
        problems=problems,
    )


def feedback_for_writer(report: GroundingReport, sources: Sequence[Source]) -> str:
    """Turn a failed check into concrete instructions for the next draft."""
    lines = ["Your draft failed the automated grounding check:"]
    lines += [f"- {problem}" for problem in report.problems]
    for block in report.uncited_blocks[:5]:
        preview = block[:_PREVIEW_CHARS] + ("..." if len(block) > _PREVIEW_CHARS else "")
        lines.append(f'- uncited: "{preview}"')
    allowed = ", ".join(s.id for s in sources if s.has_content)
    lines.append(f"Cite only these source ids, in the form [id]: {allowed}.")
    lines.append("Cite every paragraph and bullet. Remove any claim the sources do not support.")
    return "\n".join(lines)


def render_report(draft: str, sources: Sequence[Source], cited_ids: Sequence[str]) -> str:
    """Append a deterministic source list covering exactly the sources that were cited."""
    by_id = {s.id: s for s in sources}
    entries = []
    for cid in cited_ids:
        source = by_id.get(cid)
        if source is not None:
            entries.append(f"- [{source.id}] {source.title or source.url} - {source.url}")
    return f"{draft.rstrip()}\n\n## Sources\n" + "\n".join(entries) + "\n"
