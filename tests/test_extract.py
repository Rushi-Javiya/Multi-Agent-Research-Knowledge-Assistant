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
    _, text = extract_text("<body><p>Just text</p><script>bad()</script></body>", 1000)
    assert text == "Just text"


def test_truncates_on_word_boundary():
    _, text = extract_text("<body><p>alpha beta gamma delta</p></body>", 12)
    assert text == "alpha beta"


def test_empty_html_gives_empty_text():
    assert extract_text("", 100) == ("", "")
