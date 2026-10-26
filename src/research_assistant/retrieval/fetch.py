"""HTTP page fetching with SSRF protection and size limits.

URLs come from search results, so they are untrusted. Every request (including each
redirect hop) is checked so the fetcher can't be steered to localhost or a private network.
"""

import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

Resolver = Callable[..., list[Any]]

_ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_USER_AGENT = "ResearchAssistant/0.1 (+https://github.com/Rushi-Javiya/Multi-Agent-Research-Knowledge-Assistant)"


class FetchError(Exception):
    """A page could not be fetched safely."""


def is_public_host(host: str, resolver: Resolver = socket.getaddrinfo) -> bool:
    """True only if every address the host resolves to is publicly routable."""
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = resolver(host, None)
        except OSError:
            return False
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    return bool(addresses) and all(addr.is_global for addr in addresses)


class PageFetcher:
    def __init__(
        self,
        timeout: float = 15.0,
        max_bytes: int = 2_000_000,
        client: httpx.Client | None = None,
        resolver: Resolver = socket.getaddrinfo,
    ) -> None:
        self._max_bytes = max_bytes
        self._resolver = resolver
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=5,
            headers={"User-Agent": _USER_AGENT},
        )
        self._client.event_hooks["request"] = [self._guard]

    def _guard(self, request: httpx.Request) -> None:
        parts = urlsplit(str(request.url))
        if parts.scheme not in ("http", "https"):
            raise FetchError(f"blocked scheme: {parts.scheme!r}")
        if not parts.hostname or not is_public_host(parts.hostname, self._resolver):
            raise FetchError(f"blocked non-public host: {parts.hostname!r}")

    def fetch(self, url: str) -> str:
        """Return the page body as text, or raise FetchError."""
        try:
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if not content_type.startswith(_ALLOWED_TYPES):
                    raise FetchError(f"unsupported content type: {content_type or 'unknown'}")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_bytes:
                        break
                encoding = response.encoding or "utf-8"
        except FetchError:
            raise
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc
        return bytes(body[: self._max_bytes]).decode(encoding, errors="replace")

    def close(self) -> None:
        self._client.close()
