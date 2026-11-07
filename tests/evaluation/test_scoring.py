import pytest

from codelens.evaluation.scoring import file_of, precision, recall


def test_file_of() -> None:
    assert file_of("app/auth.py:15-17") == "app/auth.py"


def test_recall() -> None:
    expected = ["a.py", "b.py"]

    assert recall(expected, ["a.py:1-5", "a.py:7-9", "c.py:1-2"]) == 0.5
    assert recall(expected, ["a.py:1-5", "b.py:1-2"]) == 1.0
    assert recall(expected, []) == 0.0
    assert recall([], ["a.py:1-5"]) is None


def test_precision() -> None:
    expected = ["a.py"]

    assert precision(expected, ["a.py:1-5", "c.py:1-2"]) == pytest.approx(0.5)
    assert precision(expected, []) is None
    assert precision([], ["a.py:1-5"]) is None
