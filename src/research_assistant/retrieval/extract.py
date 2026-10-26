"""Turn raw HTML into clean main-content text."""

import re

from bs4 import BeautifulSoup

_NOISE_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe",
]
_MAIN_SELECTORS = ["article", "main", "[role=main]"]
_MIN_FALLBACK_WORDS = 4


def extract_text(html: str, max_chars: int) -> tuple[str, str]:
    """Return ``(title, text)`` with boilerplate removed and length bounded."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    main = next((node for sel in _MAIN_SELECTORS if (node := soup.select_one(sel))), None)
    root = main or soup
    lines = (re.sub(r"\s+", " ", line).strip() for line in root.get_text("\n").splitlines())
    kept = [line for line in lines if line]
    if main is None:
        # No article/main landmark: drop short fragments, which are mostly menu items.
        kept = [line for line in kept if len(line.split()) >= _MIN_FALLBACK_WORDS]
    text = "\n".join(kept)

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return title, text
