"""Which files we index, and how tree-sitter names their functions and classes."""

from pathlib import PurePosixPath

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".java": "java",
    ".cs": "csharp",
}

PLAIN_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".txt", ".rst", ".toml", ".yml", ".yaml", ".json", ".cfg", ".ini"}
)
PLAIN_TEXT_LANGUAGE = "text"

FUNCTION_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition"}),
    "javascript": frozenset({"function_declaration", "method_definition"}),
    "java": frozenset({"method_declaration", "constructor_declaration"}),
    "csharp": frozenset({"method_declaration", "constructor_declaration"}),
}

CLASS_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"class_definition"}),
    "javascript": frozenset({"class_declaration"}),
    "java": frozenset({"class_declaration", "interface_declaration"}),
    "csharp": frozenset({"class_declaration", "interface_declaration", "struct_declaration"}),
}


def detect_language(file_path: str) -> str | None:
    """Tree-sitter language name, "text" for plain text, or None if we do not index the file."""
    suffix = PurePosixPath(file_path).suffix.lower()
    if suffix in LANGUAGE_BY_EXTENSION:
        return LANGUAGE_BY_EXTENSION[suffix]
    if suffix in PLAIN_TEXT_EXTENSIONS:
        return PLAIN_TEXT_LANGUAGE
    return None
