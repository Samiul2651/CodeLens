from codelens.answering.prompts import build_prompt, cited_sources
from codelens.core.models import Chunk

LOGIN = Chunk("r", "app/auth.py", "python", 15, 17, "def login(): ...", "AuthService.login")
README = Chunk("r", "README.md", "text", 1, 3, "# Shop")


def test_prompt_has_labelled_excerpts_then_the_question() -> None:
    prompt = build_prompt([LOGIN, README], "How does login work?")

    assert '<excerpt source="app/auth.py:15-17" language="python" symbol="AuthService.login">' in (
        prompt
    )
    assert '<excerpt source="README.md:1-3" language="text">' in prompt
    assert prompt.index("<excerpt") < prompt.index("Question: How does login work?")
    assert prompt.endswith("Question: How does login work?")


def test_cited_sources_keeps_shown_order_and_drops_made_up_ones() -> None:
    shown = ["app/auth.py:15-17", "README.md:1-3", "web/cart.js:4-6"]
    text = "See [web/cart.js:4-6] and [app/auth.py:15-17]; also [made/up.py:1-2]."

    assert cited_sources(text, shown) == ["app/auth.py:15-17", "web/cart.js:4-6"]


def test_no_citations() -> None:
    assert cited_sources("I don't know.", ["a.py:1-2"]) == []
