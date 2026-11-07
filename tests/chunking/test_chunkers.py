import pytest

from codelens.chunking.chunkers import LineWindowChunker, TreeSitterChunker
from codelens.chunking.languages import detect_language
from tests.conftest import SAMPLE_REPO

AUTH_PY = (SAMPLE_REPO / "app" / "auth.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "language"),
    [("a/b.py", "python"), ("x.JS", "javascript"), ("README.md", "text"), ("logo.png", None)],
)
def test_detect_language(path: str, language: str | None) -> None:
    assert detect_language(path) == language


def test_line_windows_with_overlap() -> None:
    text = "\n".join(f"line {n}" for n in range(1, 11))

    chunks = LineWindowChunker(window=4, overlap=1).chunk_file("r", "notes.txt", text)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (4, 7), (7, 10)]
    assert chunks[0].text == "line 1\nline 2\nline 3\nline 4"
    assert all(c.symbol is None and c.repo_id == "r" for c in chunks)


@pytest.mark.parametrize(("window", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_line_window_rejects_bad_settings(window: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        LineWindowChunker(window, overlap)


def test_tree_sitter_chunks_auth_py() -> None:
    chunks = TreeSitterChunker(max_lines=8).chunk_file("r", "app/auth.py", AUTH_PY)

    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [
        (1, 4, None),  # imports + SECRET
        (7, 8, "hash_password"),
        (11, 11, None),  # class AuthService:
        (12, 13, "AuthService.__init__"),
        (15, 17, "AuthService.login"),
        (19, 21, "AuthService.is_strong"),  # starts at @staticmethod
        (23, 23, None),  # class Policy:
        (24, 25, "AuthService.Policy.max_attempts"),
        (28, 35, "audit_report"),  # 14 lines > 8: split
        (36, 41, "audit_report"),
    ]
    lines = AUTH_PY.splitlines()
    for chunk in chunks:
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.language == "python"


def test_tree_sitter_keeps_every_non_blank_line() -> None:
    covered: set[int] = set()
    for chunk in TreeSitterChunker(max_lines=8).chunk_file("r", "app/auth.py", AUTH_PY):
        covered.update(range(chunk.start_line, chunk.end_line + 1))

    assert {n for n, line in enumerate(AUTH_PY.splitlines(), 1) if line.strip()} <= covered


def test_tree_sitter_other_languages_and_text() -> None:
    chunker = TreeSitterChunker(max_lines=40)
    cs = (SAMPLE_REPO / "payments" / "PaymentService.cs").read_text(encoding="utf-8")
    readme = (SAMPLE_REPO / "README.md").read_text(encoding="utf-8")

    cs_symbols = {c.symbol for c in chunker.chunk_file("r", "payments/PaymentService.cs", cs)}
    readme_chunks = chunker.chunk_file("r", "README.md", readme)

    assert {"PaymentService.Charge", "PaymentService.Refund"} <= cs_symbols
    assert [(c.language, c.symbol) for c in readme_chunks] == [("text", None)]


def test_lone_closing_braces_are_not_chunks() -> None:
    cart = (SAMPLE_REPO / "web" / "cart.js").read_text(encoding="utf-8")

    chunks = TreeSitterChunker(max_lines=40).chunk_file("r", "web/cart.js", cart)

    assert all(any(ch.isalnum() for ch in c.text) for c in chunks)
    assert "Cart.total" in {c.symbol for c in chunks}


def test_nested_function_stays_in_parent() -> None:
    source = "def outer():\n    def inner():\n        return 1\n    return inner()\n"

    chunks = TreeSitterChunker(max_lines=50).chunk_file("r", "x.py", source)

    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [(1, 4, "outer")]
