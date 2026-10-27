"""Helpers shared by agent nodes."""

from collections.abc import Callable
from typing import Any

from research_assistant.state import ResearchState

NodeFn = Callable[[ResearchState], dict[str, Any]]


def message_text(content: Any) -> str:
    """Flatten a chat message's content (a string or a list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
