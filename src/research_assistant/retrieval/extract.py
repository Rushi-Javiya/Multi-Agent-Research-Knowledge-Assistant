"""Turn raw HTML into clean main-content text."""

import re

from bs4 import BeautifulSoup

_NOISE_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe",
]
_MAIN_SELECTORS = ["article", "main", "[role=main]"]


def extract_text(html: str, max_chars: int) -> tuple[str, str]:
    """Return ``(title, text)`` with boilerplate removed and length bounded."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    root = next((node for sel in _MAIN_SELECTORS if (node := soup.select_one(sel))), soup)
    lines = (re.sub(r"\s+", " ", line).strip() for line in root.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return title, text
