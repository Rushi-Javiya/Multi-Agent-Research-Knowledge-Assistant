from research_assistant.models import Source, normalize_url, source_id_for


def test_equivalent_urls_share_one_id():
    a = "https://Example.com/path/"
    b = "https://example.com/path#section"
    assert normalize_url(a) == normalize_url(b)
    assert source_id_for(a) == source_id_for(b)


def test_different_urls_have_different_ids():
    assert source_id_for("https://a.com/x") != source_id_for("https://a.com/y")


def test_source_from_url_sets_id_and_normalises():
    s = Source.from_url("https://Example.com/a/", title="A")
    assert s.url == "https://example.com/a"
    assert s.id == source_id_for("https://example.com/a")
    assert not s.has_content
