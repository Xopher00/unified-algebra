"""Parser for the .ua DSL. Grammar in _grammar.py, resolution in _resolver.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from unialg.assembly import compile_program
from ._resolve_cells import NamedCell as NamedCell
from ._resolver import UASpec as UASpec

if TYPE_CHECKING:
    from unialg.assembly import Program


def _source_location(text: str, remainder: str) -> tuple[int, int, str]:
    offset = len(text) - len(remainder)
    consumed = text[:offset]
    lines = consumed.split('\n')
    line_no = len(lines)
    col = len(lines[-1]) + 1
    all_lines = text.split('\n')
    source_line = all_lines[line_no - 1] if line_no <= len(all_lines) else ''
    return line_no, col, source_line


def parse_ua_spec(text: str) -> UASpec:
    """Parse .ua source text into a UASpec without compiling."""
    import hydra.parsers as P
    import hydra.parsing as HP

    from ._grammar import _build_parser
    from ._resolver import _resolve_spec

    program_parser = _build_parser()
    result = P.run_parser(program_parser, text)

    if isinstance(result, HP.ParseResultFailure):
        err = result.value
        rem = err.remainder or ''
        line_no, col, source_line = _source_location(text, rem)
        raise SyntaxError(
            f"Parse error: {err.message}",
            ("<ua source>", line_no, col, source_line)
        )

    raw_decls = result.value.value
    remainder = result.value.remainder.strip()
    if remainder:
        line_no, col, source_line = _source_location(text, remainder)
        raise SyntaxError(
            f"Unexpected input",
            ("<ua source>", line_no, col, source_line)
        )

    return _resolve_spec(raw_decls)


def parse_ua(text: str, backend=None) -> "Program":
    """Parse and compile .ua source text to a Program.

    If backend is None, uses the backend specified by ``import <name>``
    in the .ua source. A backend kwarg overrides the .ua import.
    """
    from unialg.backend import resolve_backend
    spec = parse_ua_spec(text)

    if backend is None:
        if spec.backend_name is None:
            raise ValueError(
                "No backend specified — pass backend= or add 'import <backend>' to .ua source")
        backend = resolve_backend(spec.backend_name)

    if spec.defines:
        from unialg.assembly import register_defines
        register_defines(spec.defines, backend)

    return compile_program(
        spec.equations,
        backend=backend,
        semirings=spec.semirings or None,
        cells=spec.cells or None,
        share_groups=spec.share_groups or None,
    )
