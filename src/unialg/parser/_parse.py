"""Public parse entry points for the .ua DSL."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._grammar import _source_location
from ._resolver import UASpec

if TYPE_CHECKING:
    from unialg.assembly import Program


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
        backend = register_defines(spec.defines, backend)

    from unialg.assembly import compile_program
    return compile_program(
        spec.equations,
        backend=backend,
        semirings=spec.semirings or None,
        cells=spec.cells or None,
        share_groups=spec.share_groups or None,
    )
