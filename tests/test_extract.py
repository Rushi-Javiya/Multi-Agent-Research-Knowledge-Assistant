from research_assistant.retrieval.extract import extract_text

PAGE = """
<html><head><title> My Page </title><script>var x=1;</script></head>
<body>
  <nav>Home | About</nav>
  <header>Site banner</header>
  <article><h1>Heading</h1><p>First   paragraph.</p><p>Second paragraph.</p></article>
  <footer>Copyright</footer>
</body></html>
"""


def test_extracts_title_and_main_content_only():
    title, text = extract_text(PAGE, 1000)
    assert title == "My Page"
    assert text == "Heading\nFirst paragraph.\nSecond paragraph."


def test_falls_back_to_body_without_article():
    html = "<body><p>Just some plain text</p><script>bad()</script></body>"
    _, text = extract_text(html, 1000)
    assert text == "Just some plain text"


def test_fallback_drops_short_menu_fragments():
    html = (
        "<body><div>Courses</div><div>Tutorials</div><div>Interview Prep</div>"
        "<p>LangGraph is a framework for building stateful agents.</p></body>"
    )
    _, text = extract_text(html, 1000)
    assert text == "LangGraph is a framework for building stateful agents."


def test_article_content_keeps_short_headings():
    _, text = extract_text("<article><h2>Setup</h2><p>Install it first.</p></article>", 1000)
    assert text == "Setup\nInstall it first."


def test_truncates_on_word_boundary():
    _, text = extract_text("<body><p>alpha beta gamma delta</p></body>", 12)
    assert text == "alpha beta"


def test_empty_html_gives_empty_text():
    assert extract_text("", 100) == ("", "")
