"""Parser for the .ua DSL. Grammar in _grammar.py, resolution in _resolver.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unialg.runtime import Program


@dataclass
class UASpec:
    """Parsed .ua program before compilation."""
    semirings: dict[str, Any] = field(default_factory=dict)
    sorts: dict[str, Any] = field(default_factory=dict)
    equations: list[Any] = field(default_factory=list)
    specs: list[Any] = field(default_factory=list)
    lenses: list[Any] = field(default_factory=list)


def parse_ua_spec(text: str) -> UASpec:
    """Parse .ua source text into a UASpec without compiling."""
    import hydra.parsers as P
    import hydra.parsing as HP

    from unialg.parser._grammar import _build_parser
    from unialg.parser._resolver import _resolve_spec

    program_parser = _build_parser()
    result = P.run_parser(program_parser, text)

    if isinstance(result, HP.ParseResultFailure):
        err = result.value
        snippet = repr(err.remainder[:40]) if err.remainder else "<end of input>"
        raise SyntaxError(
            f"Parse error: {err.message} at {snippet}"
        )

    raw_decls = result.value.value
    remainder = result.value.remainder.strip()
    if remainder:
        snippet = repr(remainder[:40])
        raise SyntaxError(f"Unexpected input near {snippet}")

    return _resolve_spec(raw_decls)


def parse_ua(text: str, backend) -> "Program":
    """Parse and compile .ua source text to a Program."""
    from unialg.runtime import compile_program

    spec = parse_ua_spec(text)

    return compile_program(
        spec.equations,
        backend=backend,
        specs=spec.specs,
        lenses=spec.lenses or None,
        semirings=spec.semirings or None,
    )
