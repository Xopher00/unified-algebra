"""Semiring factory and custom operation registration — structure layer.

Responsibility: bridge BackendOps (structure layer) and Semiring (semantics
layer).  This file is the only place that knows both.  semantics/semirings.py
stays free of structure-layer imports.

Layer position: structure/ — allowed to import from semantics/.  Downstream
code (user application layer) calls these functions to build Semiring objects
before passing them to contract_morphism in semantics/contractions.py.

Op name auto-derivation
───────────────────────
When a user calls semiring_from_backend with plus_op="add", the factory looks
up BOTH "add" (elementwise) and "reduce.add" (fold) in BackendOps automatically.
The user only declares the BASE op name; the system derives the reduction
variant.  The same convention applies to custom ops registered via
register_semiring_op: registering "my_op" also registers "reduce.my_op" (when
reduce_fn is provided).
"""

from __future__ import annotations

from collections.abc import Callable

from unialg.objects import Type
from unialg.semantics import Morphism
from unialg.tensors.semirings import Semiring
from unialg.emitters import BackendOps, register_backend_primitive, type_from_spec, coder_for_type


def _lookup(name: str, ops: BackendOps, label: str) -> Morphism:
    """Look up a logical op name in BackendOps, raising a clear error if absent."""
    if name not in ops:
        raise KeyError(
            f"Semiring factory: op '{name}' not found in BackendOps ({label}). "
            f"Available ops: {sorted(ops.keys())}"
        )
    return ops[name]


def semiring_from_backend(
    name: str,
    plus_op: str,
    times_op: str,
    zero: Morphism,
    one: Morphism,
    ops: BackendOps,
    *,
    adjoint_op: str | None = None,
) -> Semiring:
    """Build a Semiring by resolving op names against a BackendOps instance.

    Automatically looks up both the elementwise form and "reduce.<op>" for
    each op name, populating all reduce fields on the returned Semiring.
    Custom ops registered via register_semiring_op are resolved identically
    to built-in ops.

    Args:
        name: Human-readable semiring name (e.g. "real", "tropical").
        plus_op: Logical name of ⊕ (e.g. "add", "maximum", "minimum").
        times_op: Logical name of ⊗ (e.g. "multiply", "minimum", "add").
        zero: Morphism for the additive identity (1 → carrier).
        one: Morphism for the multiplicative identity (1 → carrier).
        ops: BackendOps instance to resolve op names against.
        adjoint_op: Logical name of the right residual of ⊗ (optional).
            E.g. "divide" for reals, "godel_implication" for fuzzy,
            "subtract" for tropical.

    Examples::

        real    = semiring_from_backend("real",    "add",     "multiply", zero, one, ops,
                                        adjoint_op="divide")
        fuzzy   = semiring_from_backend("fuzzy",   "maximum", "minimum",  zero, one, ops,
                                        adjoint_op="godel_implication")
        tropical= semiring_from_backend("tropical","minimum", "add",      zero, one, ops,
                                        adjoint_op="subtract")
    """
    plus = _lookup(plus_op, ops, "plus")
    times = _lookup(times_op, ops, "times")
    plus_reduce = _lookup(f"reduce.{plus_op}", ops, "plus_reduce")
    times_reduce = _lookup(f"reduce.{times_op}", ops, "times_reduce")

    adjoint = None
    adjoint_reduce = None
    if adjoint_op is not None:
        adjoint = _lookup(adjoint_op, ops, "adjoint")
        reduce_key = f"reduce.{adjoint_op}"
        if reduce_key in ops:
            adjoint_reduce = ops[reduce_key]
        # adjoint_reduce is intentionally optional: most adjoint ops (divide,
        # subtract, godel_implication) do not have a meaningful fold form in
        # backend specs.  op_env() uses times_reduce for adjoint-mode folding,
        # not adjoint_reduce.

    return Semiring(
        name=name,
        carrier=zero.cod,
        plus=plus,
        times=times,
        zero=zero,
        one=one,
        adjoint=adjoint,
        plus_reduce=plus_reduce,
        times_reduce=times_reduce,
        adjoint_reduce=adjoint_reduce,
    )


#### NOTE TO CLAUDE
### IMPLEMENTATION LOGIC IS INCORRECT
### USER CANNOT SPECIFICY A BACKEND. IT MAKES ZERO SENSE FOR A USER TO HAVE TO LOAD THE BACKEND MANUALLY
### BACKEND OPS MUST B SPECIFIED BY NAMED ALIAS ONLY. USER CAN COMPOSE THM FROM A MORPHISM, OR FROM ALIASES FOR REGISTERED OPS
### THE POINT IS: WE USE MORPHISMS. NOT A BASIC ASS NUMPY LIBRARY. THAT DEFEATS THE WHOLE FUCKING POIT
### NONE OF THIS CHEATING BULLCRAP

def register_semiring_op(
    name: str,
    fn: Callable,
    ops: BackendOps,
    *,
    reduce_fn: Callable | None = None,
    arity: int = 2,
    arg_type: Type | None = None,
) -> None:
    """Register a custom callable as a named semiring operation in BackendOps.

    After registration, ``name`` is usable identically to a built-in op name
    in semiring_from_backend.  If ``reduce_fn`` is provided, ``"reduce.<name>"``
    is also registered, making the op usable as a fold in contractions.

    Args:
        name: Logical op name (e.g. "softplus_plus").  Must not collide with
            an existing op in ``ops``.
        fn: Elementwise callable: (tensor, tensor) -> tensor for arity=2.
            For learned (neural) semirings, ``fn`` may be an nn.Module whose
            forward() closes over model parameters.  Weight updates are
            reflected automatically in subsequent runs without re-registering.
        ops: BackendOps instance to register into (mutated in place).
        reduce_fn: Axis-fold callable for ``fn``.  Provisional signature:
            (tensor, axis: int, seed: float) -> tensor.
            NOTE: the exact axis-reduction API is not yet finalized (see
            docs/NEXT_STEPS.md — "expand_dims/transpose not in backend specs").
            Provide None for ops intended only for elementwise use.
        arity: Number of arguments for the elementwise op.  Default 2.
        arg_type: Hydra Type for all arguments.  Defaults to FLOAT.

    Example::

        register_semiring_op("softplus_plus", softplus_binary, ops,
                             reduce_fn=softplus_reduce)
        sr = semiring_from_backend("smooth_max", "softplus_plus", "multiply",
                                   zero, one, ops)
    """
    if arg_type is None:
        arg_type = type_from_spec("FLOAT")

    coder = coder_for_type(arg_type)
    canonical = f"unialg.backend.{name}"
    bp = register_backend_primitive(canonical, fn, arg_type, arity,
                                    arg_coder=coder, result_coder=coder)
    ops.register(name, bp)

    if reduce_fn is not None:
        reduce_canonical = f"unialg.backend.reduce.{name}"
        # Reduce ops take a single tensor argument; the fold axis and seed are
        # encoded in the callable via closure.  Arity=1 here reflects the
        # single-tensor input after the axis/seed are bound.
        reduce_bp = register_backend_primitive(
            reduce_canonical, reduce_fn, arg_type, 1,
            arg_coder=coder, result_coder=coder,
        )
        ops.register(f"reduce.{name}", reduce_bp)
