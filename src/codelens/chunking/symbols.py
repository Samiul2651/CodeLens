"""Find functions and methods with tree-sitter."""

from dataclasses import dataclass

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from codelens.chunking.languages import CLASS_NODE_TYPES, FUNCTION_NODE_TYPES


@dataclass(frozen=True)
class Symbol:
    name: str  # qualified: "AuthService.Policy.max_attempts"
    start_line: int  # 1-based, includes Python decorators
    end_line: int


def _name(node: Node) -> str:
    name = node.child_by_field_name("name")
    return name.text.decode() if name is not None and name.text is not None else "?"


def find_functions(source: str, language: str) -> list[Symbol]:
    function_types = FUNCTION_NODE_TYPES[language]
    class_types = CLASS_NODE_TYPES[language]
    found: list[Symbol] = []

    def walk(node: Node, scope: list[str]) -> None:
        if node.type in function_types:
            start = node.start_point.row
            if node.parent is not None and node.parent.type == "decorated_definition":
                start = node.parent.start_point.row
            found.append(Symbol(".".join([*scope, _name(node)]), start + 1, node.end_point.row + 1))
        inner = [*scope, _name(node)] if node.type in class_types else scope
        for child in node.named_children:
            walk(child, inner)

    tree = get_parser(language).parse(source.encode("utf-8"))
    walk(tree.root_node, [])
    return sorted(found, key=lambda s: s.start_line)


def outermost(symbols: list[Symbol]) -> list[Symbol]:
    """Drop functions nested inside another function: they are part of the parent's chunk."""
    kept: list[Symbol] = []
    for symbol in symbols:  # sorted by start_line
        parent = kept[-1] if kept else None
        if parent and symbol.start_line >= parent.start_line and symbol.end_line <= parent.end_line:
            continue
        kept.append(symbol)
    return kept
