"""Architecture rule: codelens.core imports only the standard library and itself.

If this fails, a concrete library (Qdrant, Anthropic, FastAPI...) leaked into the core,
and the interfaces are no longer swappable.
"""

import ast
import sys
from pathlib import Path

CORE = Path(__file__).parents[2] / "src" / "codelens" / "core"


def imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_core_uses_only_stdlib_and_itself() -> None:
    allowed = set(sys.stdlib_module_names) | {"codelens"}
    files = list(CORE.glob("*.py"))
    assert files, "core package not found"

    for path in files:
        outside = imported_top_level_modules(path) - allowed
        assert not outside, f"{path.name} imports {sorted(outside)}"
