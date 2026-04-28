"""Cell interpreter — compiles a Cell expression to a runnable morphism.

Para 1-cells (eq, lit, seq, par, copy, delete, algebra_hom) compile to a
single ``Callable``. Optic 1-cells (lens) compile to a ``CompiledLens``
carrying forward and backward callables. Sequential composition (``seq``)
and monoidal product (``par``) dispatch on the children's compiled type:

  * Para children → ordinary composition / pair-bimap.
  * Optic children → Optic 2-category composition; if both children are
    height-2 (carry a residual), residuals are threaded.
  * Mixing Para and Optic children is rejected.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

from hydra.lib import pairs as _pairs

from unialg.assembly._para import Cell
from unialg.assembly._para_alg_hom import compile_algebra_hom


# ---------------------------------------------------------------------------
# Compiled morphism types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CompiledLens:
    """Compiled Optic 1-cell.

    ``residual_sort=None`` ⇒ height-1 lens (forward: A→B, backward: B→A).
    ``residual_sort=R``    ⇒ height-2 optic (forward: A→R×B, backward: R×B'→A').
    """
    forward: Callable
    backward: Callable
    residual_sort: object | None = None


CompiledMorphism: TypeAlias = Callable | CompiledLens
Matcher: TypeAlias = Callable[[object], tuple[int, list, list]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compile_cell(
    cell: Cell,
    native_fns: dict,
    coder,
    backend,
    *,
    matchers: dict[str, Matcher] | None = None,
) -> CompiledMorphism | None:
    """Compile a Cell into a Python callable (Para) or a CompiledLens (Optic).

    Returns ``None`` if any referenced Equation is missing from ``native_fns``.
    """
    return _compile(cell, native_fns, coder, backend, matchers or {})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _compile(cell, native_fns, coder, backend, matchers) -> CompiledMorphism | None:
    k = cell.kind
    if k == "eq":
        from hydra.core import Name
        return native_fns.get(Name(f"ua.equation.{cell.equation_name}"))
    if k == "lit":
        return _compile_lit(cell, coder)
    if k == "seq":
        return _compile_binary(cell, native_fns, coder, backend, matchers, _seq)
    if k == "par":
        return _compile_binary(cell, native_fns, coder, backend, matchers, _par)
    if k == "copy":
        return backend.compile(_diag)
    if k == "delete":
        return backend.compile(_unit)
    if k == "iden":
        return backend.compile(_id)
    if k == "lens":
        return _compile_lens(cell, native_fns, coder, backend, matchers)
    if k == "algebraHom":
        return compile_algebra_hom(
            cell, backend, matchers,
            compile_subcell=partial(_compile,
                                    native_fns=native_fns, coder=coder,
                                    backend=backend, matchers=matchers),
        )
    raise ValueError(f"unknown Cell kind {k!r}")


# ---------------------------------------------------------------------------
# Variant-specific compilation
# ---------------------------------------------------------------------------

def _compile_lit(cell: Cell, coder) -> Callable | None:
    from unialg.assembly.compositions import _decode_init
    value, _ = _decode_init(coder, cell.value_term)
    if value is None:
        return None
    return _const(value)


def _compile_lens(cell: Cell, native_fns, coder, backend, matchers) -> CompiledLens | None:
    fwd = _compile(cell.forward,  native_fns, coder, backend, matchers)
    bwd = _compile(cell.backward, native_fns, coder, backend, matchers)
    if fwd is None or bwd is None:
        return None
    if isinstance(fwd, CompiledLens) or isinstance(bwd, CompiledLens):
        raise ValueError(
            f"lens: forward and backward must be Para 1-cells, "
            f"got fwd={type(fwd).__name__} bwd={type(bwd).__name__}"
        )
    return CompiledLens(forward=fwd, backward=bwd, residual_sort=cell.residual_sort)


def _compile_binary(cell, native_fns, coder, backend, matchers, builder):
    f = _compile(cell.left,  native_fns, coder, backend, matchers)
    g = _compile(cell.right, native_fns, coder, backend, matchers)
    if f is None or g is None:
        return None
    return builder(f, g, backend)


# ---------------------------------------------------------------------------
# Para / Optic primitives
# ---------------------------------------------------------------------------

def _diag(x):
    return (x, x)


def _id(x):
    return x


def _unit(*_args):
    return None


def _const(v):
    return lambda *_args, _v=v: _v


def _compose(f: Callable, g: Callable, x):
    return g(f(x))


# ---------------------------------------------------------------------------
# Sequential composition (`;`) — Para + Optic dispatch
# ---------------------------------------------------------------------------

def _seq(f: CompiledMorphism, g: CompiledMorphism, backend) -> CompiledMorphism:
    f_lens = isinstance(f, CompiledLens)
    g_lens = isinstance(g, CompiledLens)
    if f_lens and g_lens:
        return _seq_optic(f, g, backend)
    if f_lens != g_lens:
        raise ValueError(_mismatch_msg("seq", f_lens, g_lens))
    return backend.compile(partial(_compose, f, g))


def _seq_optic(f: CompiledLens, g: CompiledLens, backend) -> CompiledLens:
    f_h2 = f.residual_sort is not None
    g_h2 = g.residual_sort is not None
    if f_h2 != g_h2:
        raise ValueError("seq: cannot compose height-1 lens with height-2 optic")
    if not f_h2:
        return CompiledLens(
            forward =backend.compile(partial(_compose, f.forward,  g.forward)),
            backward=backend.compile(partial(_compose, g.backward, f.backward)),
        )
    return _seq_optic_h2(f, g, backend)


def _seq_optic_h2(f: CompiledLens, g: CompiledLens, backend) -> CompiledLens:
    """Height-2 optic seq: forward produces nested residuals; backward
    consumes them and threads through both inner backwards."""
    from unialg.algebra.sort import ProductSort

    def fwd(a, _f=f, _g=g):
        r1, b = _f.forward(a)
        r2, c = _g.forward(b)
        return ((r1, r2), c)

    def bwd(payload, _f=f, _g=g):
        (r1, r2), c_prime = payload
        b_prime = _g.backward((r2, c_prime))
        return _f.backward((r1, b_prime))

    return CompiledLens(
        forward=backend.compile(fwd),
        backward=backend.compile(bwd),
        residual_sort=ProductSort([f.residual_sort, g.residual_sort]),
    )


# ---------------------------------------------------------------------------
# Monoidal product (`⊗`) — Para + Optic dispatch
# ---------------------------------------------------------------------------

def _par(f: CompiledMorphism, g: CompiledMorphism, backend) -> CompiledMorphism:
    f_lens = isinstance(f, CompiledLens)
    g_lens = isinstance(g, CompiledLens)
    if f_lens and g_lens:
        return _par_optic(f, g, backend)
    if f_lens != g_lens:
        raise ValueError(_mismatch_msg("par", f_lens, g_lens))
    return backend.compile(partial(_pairs.bimap, f, g))


def _par_optic(f: CompiledLens, g: CompiledLens, backend) -> CompiledLens:
    f_h2 = f.residual_sort is not None
    g_h2 = g.residual_sort is not None
    if f_h2 != g_h2:
        raise ValueError("par: cannot pair height-1 lens with height-2 optic")
    if not f_h2:
        return CompiledLens(
            forward =backend.compile(partial(_pairs.bimap, f.forward,  g.forward)),
            backward=backend.compile(partial(_pairs.bimap, f.backward, g.backward)),
        )
    return _par_optic_h2(f, g, backend)


def _par_optic_h2(f: CompiledLens, g: CompiledLens, backend) -> CompiledLens:
    """Height-2 optic par: residuals package into a single pair (R₁, R₂);
    forward maps (A,C) → (R₁×R₂, B×D); backward inverts."""
    from unialg.algebra.sort import ProductSort

    def fwd(p, _f=f, _g=g):
        a, c = p
        r1, b = _f.forward(a)
        r2, d = _g.forward(c)
        return ((r1, r2), (b, d))

    def bwd(payload, _f=f, _g=g):
        (r1, r2), (b_prime, d_prime) = payload
        return (_f.backward((r1, b_prime)), _g.backward((r2, d_prime)))

    return CompiledLens(
        forward=backend.compile(fwd),
        backward=backend.compile(bwd),
        residual_sort=ProductSort([f.residual_sort, g.residual_sort]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mismatch_msg(op: str, f_lens: bool, g_lens: bool) -> str:
    return (
        f"{op}: cannot mix Para and Optic 1-cells "
        f"(left is {'Optic' if f_lens else 'Para'}, "
        f"right is {'Optic' if g_lens else 'Para'})"
    )
