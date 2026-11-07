import pytest

from codelens.answering.context_packer import ContextPacker, file_priority
from codelens.core.models import Chunk
from tests.fakes import CharTokenCounter


def chunk(path: str, start: int, text: str = "x" * 300) -> Chunk:
    return Chunk("r", path, "python", start, start + 9, text)


REPO = [
    chunk("tests/test_auth.py", 1),
    chunk("src/auth.py", 11),
    chunk("src/auth.py", 1),
    chunk("README.md", 1),
    chunk("pyproject.toml", 1),
    chunk("src/cart.py", 1),
]


@pytest.mark.parametrize(
    ("path", "priority"),
    [
        ("README.md", 0),
        ("docs/README.md", 3),
        ("src/auth.py", 1),
        ("tests/test_auth.py", 2),
        ("src/test_utils.py", 2),
        ("pkg/auth_test.py", 2),
        ("pyproject.toml", 3),
    ],
)
def test_file_priority(path: str, priority: int) -> None:
    assert file_priority(path) == priority


def test_everything_fits() -> None:
    packed = ContextPacker(CharTokenCounter(), token_budget=100_000).pack(REPO, "Q?")

    assert not packed.truncated
    assert (packed.files_included, packed.files_total) == (5, 5)
    files_in_order = list(dict.fromkeys(c.file_path for c in packed.chunks))
    assert files_in_order == [
        "README.md", "src/auth.py", "src/cart.py", "tests/test_auth.py", "pyproject.toml",
    ]  # fmt: skip
    auth_starts = [c.start_line for c in packed.chunks if c.file_path == "src/auth.py"]
    assert auth_starts == [1, 11], "chunks of a file stay in line order"


def test_small_budget_keeps_whole_files_in_priority_order() -> None:
    counter = CharTokenCounter()

    packed = ContextPacker(counter, token_budget=600).pack(REPO, "Q?")

    assert packed.truncated
    files = {c.file_path for c in packed.chunks}
    assert "README.md" in files
    assert "tests/test_auth.py" not in files
    if "src/auth.py" in files:  # a file is all in or all out
        assert len([c for c in packed.chunks if c.file_path == "src/auth.py"]) == 2
    assert packed.input_tokens <= 600
    assert counter.calls == 1, "one exact count when the estimate holds"


def test_shrinks_when_the_estimate_was_too_low() -> None:
    dense = CharTokenCounter(chars_per_token=1.0)  # 3x more tokens than the packer estimates

    packed = ContextPacker(dense, token_budget=1_500).pack(REPO, "Q?")

    assert packed.input_tokens <= 1_500
    assert dense.calls >= 2
    assert packed.chunks[0].file_path == "README.md"


def test_nothing_fits() -> None:
    packed = ContextPacker(CharTokenCounter(), token_budget=10).pack(REPO, "Q?")

    assert packed.chunks == [] and packed.context == ""
    assert packed.truncated


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ContextPacker(CharTokenCounter(), token_budget=0)
