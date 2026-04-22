"""Text parser for the unified-algebra DSL (.ua files).

Layer role: syntactic sugar over the stable Python API.  Parses `.ua` source
text into the same Python objects that hand-written code would produce, then
delegates to compile_program() for compilation.

Grammar rules live in _grammar.py; name resolution lives in _resolver.py.

Entry points:
    parse_ua_spec(text)          -> UASpec  (introspection / testing)
    parse_ua(text, backend)      -> Program (ready to call)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..assembly.program import Program


# ---------------------------------------------------------------------------
# UASpec — the parse tree after resolution
# ---------------------------------------------------------------------------


@dataclass
class UASpec:
    """Parsed .ua program before compilation.

    Fields contain the resolved DSL objects (semiring terms, sort terms, etc.)
    in declaration order.  Suitable for passing directly to compile_program().
    """
    semirings: dict[str, Any] = field(default_factory=dict)   # name -> semiring Term
    sorts: dict[str, Any] = field(default_factory=dict)        # name -> sort Term
    equations: list[Any] = field(default_factory=list)         # equation Terms
    specs: list[Any] = field(default_factory=list)             # PathSpec|FanSpec|FoldSpec|…
    lenses: list[Any] = field(default_factory=list)            # lens Terms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ua_spec(text: str) -> UASpec:
    """Parse .ua source text and return a UASpec without compiling.

    Useful for introspection and testing without needing a backend.

    Args:
        text: .ua source text

    Returns:
        UASpec with semirings, sorts, equations, specs, and lenses populated.

    Raises:
        SyntaxError: if the text cannot be parsed
        ValueError:  if sort/semiring references are invalid
    """
    import hydra.parsers as P
    import hydra.parsing as HP

    from ._grammar import _build_parser
    from ._resolver import _resolve_spec

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
    """Parse a .ua program text and compile it to a Program.

    Args:
        text:    .ua source text
        backend: Backend (numpy_backend() or pytorch_backend())

    Returns:
        A compiled Program, callable by entry point name.

    Raises:
        SyntaxError: if the text cannot be parsed
        ValueError:  if sort junctions or references are invalid
    """
    from ..assembly.program import compile_program

    spec = parse_ua_spec(text)

    return compile_program(
        spec.equations,
        backend=backend,
        specs=spec.specs,
        lenses=spec.lenses if spec.lenses else None,
        semirings=spec.semirings if spec.semirings else None,
    )
