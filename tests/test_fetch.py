import httpx
import pytest

from research_assistant.retrieval.fetch import FetchError, PageFetcher, is_public_host


def public_resolver(host, port):
    return [(2, 1, 6, "", ("93.184.216.34", 0))]


def fetcher_for(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return PageFetcher(client=client, resolver=public_resolver, **kwargs)


def html_ok(request):
    return httpx.Response(200, text="<p>hello</p>", headers={"content-type": "text/html"})


def test_fetches_html():
    assert fetcher_for(html_ok).fetch("https://example.com") == "<p>hello</p>"


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"]
)
def test_private_ip_literals_are_not_public(host):
    assert not is_public_host(host)


def test_public_ip_literal_is_public():
    assert is_public_host("93.184.216.34")


def test_hostname_resolving_to_private_address_is_blocked():
    private = lambda host, port: [(2, 1, 6, "", ("10.0.0.7", 0))]  # noqa: E731
    assert not is_public_host("internal.corp", private)


def test_unresolvable_host_is_blocked():
    def fail(host, port):
        raise OSError("no such host")

    assert not is_public_host("nope.invalid", fail)


def test_localhost_url_is_blocked_without_making_a_request():
    calls = []

    def handler(request):
        calls.append(request)
        return html_ok(request)

    with pytest.raises(FetchError, match="non-public"):
        fetcher_for(handler).fetch("http://127.0.0.1:8080/admin")
    assert calls == []


def test_non_http_scheme_is_blocked():
    with pytest.raises(FetchError, match="scheme"):
        fetcher_for(html_ok).fetch("file:///etc/passwd")


def test_redirect_to_private_host_is_blocked():
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})
        return html_ok(request)

    with pytest.raises(FetchError, match="non-public"):
        fetcher_for(handler).fetch("https://example.com/start")


def test_http_error_becomes_fetch_error():
    handler = lambda request: httpx.Response(404)  # noqa: E731
    with pytest.raises(FetchError, match="404"):
        fetcher_for(handler).fetch("https://example.com/missing")


def test_non_text_content_type_rejected():
    handler = lambda request: httpx.Response(  # noqa: E731
        200, content=b"%PDF", headers={"content-type": "application/pdf"}
    )
    with pytest.raises(FetchError, match="content type"):
        fetcher_for(handler).fetch("https://example.com/a.pdf")


def test_body_is_capped():
    handler = lambda request: httpx.Response(  # noqa: E731
        200, content=b"x" * 5000, headers={"content-type": "text/plain"}
    )
    assert len(fetcher_for(handler, max_bytes=100).fetch("https://example.com")) == 100
