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
    backend_name: str | None = None


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


_BACKEND_MAP = {
    'numpy': 'NumpyBackend',
    'jax': 'JaxBackend',
    'pytorch': 'PytorchBackend',
    'cupy': 'CupyBackend',
}


def _resolve_backend(name: str):
    cls_name = _BACKEND_MAP.get(name)
    if cls_name is None:
        raise ValueError(
            f"Unknown backend {name!r} — available: {list(_BACKEND_MAP)}")
    import unialg.backend as be
    return getattr(be, cls_name)()


def parse_ua(text: str, backend=None) -> "Program":
    """Parse and compile .ua source text to a Program.

    If backend is None, uses the backend specified by ``import <name>``
    in the .ua source. A backend kwarg overrides the .ua import.
    """
    from unialg.runtime import compile_program

    spec = parse_ua_spec(text)

    if backend is None:
        if spec.backend_name is None:
            raise ValueError(
                "No backend specified — pass backend= or add 'import <backend>' to .ua source")
        backend = _resolve_backend(spec.backend_name)

    return compile_program(
        spec.equations,
        backend=backend,
        specs=spec.specs,
        lenses=spec.lenses or None,
        semirings=spec.semirings or None,
    )
