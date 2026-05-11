"""Backend primitive registration for the unialg DSL.

Translates backend JSON specs into Hydra primitives and morphisms.

Pipeline::

    JSON spec
      -> load_spec             (resolve paths, look up types and codecs)
      -> register_backend_primitive  (build Hydra Primitive with typed impl)
      -> BackendOps            (wrap all primitives as Morphism objects)

Canonical names follow the pattern ``unialg.backend.<op>``
(e.g. ``unialg.backend.add``, ``unialg.backend.reduce.add``).
The same name is used regardless of which library provides the implementation,
so DSL code is backend-agnostic.

Spec format (JSON)::

    {
      "backend": "numpy",
      "operations": {
        "add": {
          "kind":        "elementwise binary",
          "path":        "numpy.add",
          "arity":       2,
          "arg_type":    "FLOAT",
          "result_type": "FLOAT",
          "codec":       "float64"
        }
      }
    }
"""

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hydra.core import LiteralType, Name, Type, TypeLiteral
from hydra.graph import Primitive
from hydra.dsl.python import Right
import hydra.dsl.meta.phantoms as P

from unialg.objects import ExpType, TypeScheme, ProductType
from unialg.semantics import morphisms
from unialg.syntax import expressions as expr


TYPE_REGISTRY: dict[str, Type] = {
    "INT":   TypeLiteral(LiteralType.INTEGER),
    "FLOAT": TypeLiteral(LiteralType.FLOAT),
}

# Each entry: (unwrap: Term -> Python, wrap: Python -> Term)
CODEC_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "int32":   (lambda t: t.value.value.value,   lambda x: P.int32(int(x)).value),
    "int64":   (lambda t: t.value.value.value,   lambda x: P.int64(int(x)).value),
    "float32": (lambda t: t.value.value.value,   lambda x: P.float32(float(x)).value),
    "float64": (lambda t: t.value.value.value,   lambda x: P.float64(float(x)).value),
}

_CANONICAL_PREFIX = "unialg.backend"


@dataclass(frozen=True)
class BackendPrimitive:
    """A resolved backend operation ready for Hydra primitive registration.

    Carries the Hydra ``Primitive`` (name + type scheme + impl), the arity, and
    the Hydra types for arguments and result.  Produced by
    ``register_backend_primitive``; consumed by ``_primitive_morphism`` and
    ``BackendOps``.
    """

    primitive: Primitive
    arity: int
    arg_type: object
    result_type: object

    @property
    def name(self) -> Name:
        """Canonical Hydra name, e.g. ``Name("unialg.backend.add")``."""
        return self.primitive.name


def resolve_function(path: str):
    """Import and return a callable by dotted path.

    The last segment is the attribute name; everything before it is the module::

        resolve_function("numpy.add")           -> numpy.add
        resolve_function("jax.numpy.add")       -> jax.numpy.add
        resolve_function("scipy.special.expit") -> scipy.special.expit
    """
    module_name, function_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), function_name)


def infer_arity(fn) -> int:
    """Return the number of required positional parameters of ``fn``.

    Checks ``fn.nin`` first (NumPy/JAX ufunc convention), then falls back to
    ``inspect.signature``.  Not used by ``load_spec`` — specs must carry an
    explicit ``arity`` field — but available for ad-hoc registration.
    """
    if hasattr(fn, "nin"):
        return fn.nin
    sig = inspect.signature(fn)
    return sum(
        1 for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _curried_type(arg_type, result_type, arity: int):
    """Build the curried Hydra type ``arg_type -> ... -> result_type`` for ``arity`` arguments."""
    typ = result_type
    for _ in range(arity):
        typ = ExpType(arg_type, typ)
    return typ


def repeated_product(t, n):
    """Build the left-nested product type ``t × t × ... × t`` (``n`` copies).

    ``n=1`` returns ``t`` directly; ``n=2`` returns ``ProductType(t, t)``, etc.
    Used to construct the visible domain type for a morphism of arity ``n``.
    """
    if n == 1:
        return t
    out = t
    for _ in range(n - 1):
        out = ProductType(out, t)
    return out


def product_arg(x, n):
    """Destructure a left-nested Hydra pair term into a list of ``n`` component terms.

    Mirrors the shape produced by ``repeated_product``: the first ``n-1``
    components are extracted with successive ``fst`` projections; the last
    component is the final ``snd``.
    """
    if n == 1:
        return [x]
    vals = []
    cur = x
    for _ in range(n - 1):
        vals.append(P.first(cur))
        cur = P.second(cur)
    vals.append(cur)
    return vals


def register_backend_primitive(
    canonical_name: str,
    path: str,
    arg_type,
    arity: int,
    *,
    unwrap: Callable,
    wrap: Callable,
    result_type=None,
) -> BackendPrimitive:
    """Register a single backend function as a Hydra primitive.

    Args:
        canonical_name: Hydra primitive name, e.g. ``"unialg.backend.add"``.
            This name is the same across all backends for the same logical op.
        path: Dotted import path to the backend function, e.g. ``"numpy.add"``.
        arg_type: Hydra ``Type`` for all input arguments (homogeneous).
        arity: Number of arguments the function takes.  Must be provided
            explicitly — do not rely on ``infer_arity`` for arbitrary APIs.
        unwrap: Converts a Hydra term to a Python value before calling ``fn``.
        wrap: Converts the Python result back to a raw Hydra term.
        result_type: Hydra ``Type`` for the return value.  Defaults to
            ``arg_type`` when omitted.

    Returns:
        A ``BackendPrimitive`` whose ``primitive`` field is registered under
        ``canonical_name`` with the appropriate curried ``TypeScheme``.
    """
    fn = resolve_function(path)
    result_type = result_type or arg_type
    name = Name(canonical_name)
    scheme = TypeScheme(
        variables=(),
        body=_curried_type(arg_type, result_type, arity),
        constraints=None,
    )

    def impl(_ctx, _graph, args):
        py_args = [unwrap(a) for a in args]
        return Right(wrap(fn(*py_args)))

    return BackendPrimitive(
        primitive=Primitive(name, scheme, impl),
        arity=arity,
        arg_type=arg_type,
        result_type=result_type,
    )


def _primitive_morphism(bp: BackendPrimitive) -> morphisms.Morphism:
    """Wrap a ``BackendPrimitive`` as a ``Morphism``.

    Builds a lambda over the product domain that applies the primitive in
    curried form, carrying the primitive in ``aux_primitives`` so ``run``
    can register it in a temporary graph automatically.
    """
    x = P.var("x")
    args = product_arg(x, bp.arity)
    term = P.primitive(bp.name)
    for arg in args:
        term = P.apply(term, arg)
    raw = P.lam("x", term).value
    dom = repeated_product(bp.arg_type, bp.arity)
    return morphisms.Morphism(
        expr.Prim(raw, dom, bp.result_type),
        aux_primitives=(bp.primitive,),
    )


def load_spec(spec: dict | str | Path) -> dict[str, BackendPrimitive]:
    """Load a backend spec and return a dict of ``BackendPrimitive`` objects.

    ``spec`` may be a parsed dict, a file path, or a path string.  Each entry
    in ``spec["operations"]`` becomes one ``BackendPrimitive`` keyed by its
    logical op name (e.g. ``"add"``, ``"reduce.add"``).

    The canonical Hydra name is derived as ``unialg.backend.<op_name>``.
    """
    if isinstance(spec, (str, Path)):
        with open(spec) as f:
            spec = json.load(f)
    result: dict[str, BackendPrimitive] = {}
    for op_name, entry in spec["operations"].items():
        arg_type = TYPE_REGISTRY[entry["arg_type"]]
        result_type = TYPE_REGISTRY[entry.get("result_type", entry["arg_type"])]
        unwrap, wrap = CODEC_REGISTRY[entry["codec"]]
        canonical = f"{_CANONICAL_PREFIX}.{op_name}"
        result[op_name] = register_backend_primitive(
            canonical, entry["path"], arg_type, entry["arity"],
            unwrap=unwrap, wrap=wrap,
            result_type=result_type,
        )
    return result


class BackendOps:
    """All morphisms for a backend, keyed by logical op name.

    Canonical Hydra names follow ``unialg.backend.<op>``
    (e.g. ``unialg.backend.add``, ``unialg.backend.reduce.add``).
    Access morphisms by their short spec key: ``ops["add"]``, ``ops["reduce.add"]``.

    Morphisms are built eagerly at construction time.  ``run`` automatically
    includes ``aux_primitives``, so no manual graph extension is needed.

    Example::

        ops = BackendOps.from_spec("backends/numpy.json")
        add = ops["add"]   # Morphism: FLOAT × FLOAT → FLOAT
    """

    def __init__(self, primitives: dict[str, BackendPrimitive]):
        self._primitives = primitives
        self._morphisms: dict[str, morphisms.Morphism] = {
            name: _primitive_morphism(bp) for name, bp in primitives.items()
        }

    @classmethod
    def from_spec(cls, spec: dict | str | Path) -> "BackendOps":
        """Construct from a JSON spec file path, path string, or parsed dict."""
        return cls(load_spec(spec))

    def __getitem__(self, name: str) -> morphisms.Morphism:
        """Return the morphism for logical op ``name`` (e.g. ``"add"``)."""
        return self._morphisms[name]

    def __contains__(self, name: str) -> bool:
        return name in self._morphisms

    def keys(self):
        """Return all registered logical op names."""
        return self._morphisms.keys()
