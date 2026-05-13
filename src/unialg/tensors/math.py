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

from hydra.context import Context
from hydra.core import LiteralType, Name, Term, Type, TypeLiteral
from hydra.dsl.python import Left, Right
from hydra.graph import Graph, Primitive, TermCoder
from hydra.lib import maps as Maps
from hydra.packaging import Library, Namespace
import hydra.dsl.meta.phantoms as P

from unialg.objects import ExpType, TypeScheme, ProductType
from unialg.semantics import morphisms
from unialg.structure import terms as struct_terms
from unialg.syntax import expressions as expr

from hydra.lib import maps as Maps

TYPE_REGISTRY: dict[str, Type] = {
    "INT":   TypeLiteral(LiteralType.INTEGER),
    "FLOAT": TypeLiteral(LiteralType.FLOAT),
}

# Each entry: (unwrap: Term -> Python, wrap: Python -> Term)
def _expect_right(result, context: str):
    """Unwrap a Hydra Either result or raise a readable error."""
    if isinstance(result, Left):
        raise TypeError(f"{context}: {result.value!r}")
    return result.value


def _literal_value(term: Term, context: str):
    """Extract the Python literal payload from a Hydra literal term."""
    try:
        return term.value.value.value
    except Exception as e:
        raise TypeError(f"{context}: expected literal term, got {term!r}") from e


def _mk_term_coder(
    typ: Type,
    decode_term: Callable[[Term], object],
    encode_value: Callable[[object], Term],
) -> TermCoder:
    """Construct a Hydra TermCoder from native decode/encode callables."""
    return TermCoder(
        type=typ,
        encode=lambda _cx, _graph, term: Right(decode_term(term)),
        decode=lambda _cx, value: Right(encode_value(value)),
    )


TERM_CODER_REGISTRY: dict[str, TermCoder] = {
    "int32": _mk_term_coder(
        TypeLiteral(LiteralType.INTEGER),
        lambda t: int(_literal_value(t, "int32 coder")),
        lambda x: P.int32(int(x)).value,
    ),
    "int64": _mk_term_coder(
        TypeLiteral(LiteralType.INTEGER),
        lambda t: int(_literal_value(t, "int64 coder")),
        lambda x: P.int64(int(x)).value,
    ),
    "float32": _mk_term_coder(
        TypeLiteral(LiteralType.FLOAT),
        lambda t: float(_literal_value(t, "float32 coder")),
        lambda x: P.float32(float(x)).value,
    ),
    "float64": _mk_term_coder(
        TypeLiteral(LiteralType.FLOAT),
        lambda t: float(_literal_value(t, "float64 coder")),
        lambda x: P.float64(float(x)).value,
    ),
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
    arg_type: Type
    result_type: Type
    arg_coder: TermCoder
    result_coder: TermCoder

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
    arg_type: Type,
    arity: int,
    *,
    arg_coder: TermCoder,
    result_coder: TermCoder,
    result_type: Type | None = None,
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

    def impl(ctx: Context, graph: Graph, args):
        py_args = [
            _expect_right(
                arg_coder.encode(ctx, graph, arg),
                f"decoding argument for {canonical_name}",
            )
            for arg in args
        ]
        py_result = fn(*py_args)
        return result_coder.decode(ctx, py_result)

    return BackendPrimitive(
        primitive=Primitive(name, scheme, impl),
        arity=arity,
        arg_type=arg_type,
        result_type=result_type,
        arg_coder=arg_coder,
        result_coder=result_coder,
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
    raw = struct_terms.normalize_term(P.lam("x", term)).value
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
        coder = TERM_CODER_REGISTRY[entry["codec"]]
        canonical = f"{_CANONICAL_PREFIX}.{op_name}"
        result[op_name] = register_backend_primitive(
            canonical,
            entry["path"],
            arg_type,
            entry["arity"],
            arg_coder=coder,
            result_coder=coder,
            result_type=result_type,
        )
    return result


def backend_library(primitives: dict[str, BackendPrimitive]) -> Library:
    """Build a Hydra Library from registered backend primitives."""
    return Library(
        namespace=Namespace(_CANONICAL_PREFIX),
        prefix="backend",
        primitives=tuple(bp.primitive for bp in primitives.values()),
    )


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
        self._library = backend_library(primitives)
        self._morphisms: dict[str, morphisms.Morphism] = {
            name: _primitive_morphism(bp) for name, bp in primitives.items()
        }

    @classmethod
    def from_spec(cls, spec: dict | str | Path) -> "BackendOps":
        """Construct from a JSON spec file path, path string, or parsed dict."""
        return cls(load_spec(spec))

    @property
    def library(self) -> Library:
        """The Hydra-native primitive library for this backend."""
        return self._library

    @property
    def primitives(self) -> dict[str, BackendPrimitive]:
        """The registered backend primitives keyed by logical op name."""
        return self._primitives

    def __getitem__(self, name: str) -> morphisms.Morphism:
        """Return the morphism for logical op ``name`` (e.g. ``"add"``)."""
        return self._morphisms[name]

    def __contains__(self, name: str) -> bool:
        return name in self._morphisms

    def keys(self):
        """Return all registered logical op names."""
        return self._morphisms.keys()

    @property
    def op_names(self) -> frozenset[str]:
        """Logical operation names supported by this backend."""
        return frozenset(self._morphisms.keys())

    @property
    def hydra_names(self) -> frozenset[Name]:
        """Canonical Hydra primitive names supported by this backend."""
        return frozenset(bp.name for bp in self._primitives.values())

    def to_graph(self, base: Graph | None = None) -> Graph:
        """Install this backend's primitives into a Hydra Graph."""
        return library_to_graph(self._library, base)

    def coverage(self, required: set[str] | frozenset[str]) -> dict[str, frozenset[str]]:
        """Report supported and missing logical ops for this backend."""
        return backend_coverage(self, required)



def library_primitives_map(library: Library):
    """Convert a Hydra Library into a Graph.primitives-style map."""
    return Maps.from_dict({prim.name: prim for prim in library.primitives})


def library_to_graph(library: Library, base: Graph | None = None) -> Graph:
    """Install a backend Library's primitives into a Hydra Graph."""
    base = base or Graph(
        bound_terms=Maps.empty(),
        bound_types=Maps.empty(),
        class_constraints=Maps.empty(),
        lambda_variables=frozenset(),
        metadata=Maps.empty(),
        primitives=Maps.empty(),
        schema_types=Maps.empty(),
        type_variables=frozenset(),
    )

    prims = dict(base.primitives)
    for prim in library.primitives:
        prims[prim.name] = prim

    return Graph(
        bound_terms=base.bound_terms,
        bound_types=base.bound_types,
        class_constraints=base.class_constraints,
        lambda_variables=base.lambda_variables,
        metadata=base.metadata,
        primitives=Maps.from_dict(prims),
        schema_types=base.schema_types,
        type_variables=base.type_variables,
    )


def backend_graph(ops: "BackendOps", base: Graph | None = None) -> Graph:
    """Create a Hydra Graph containing all primitives for a backend."""
    return library_to_graph(ops.library, base)


def backend_op_names(ops: "BackendOps") -> frozenset[str]:
    """Return the logical op names supported by a backend."""
    return frozenset(ops.keys())


def backend_hydra_names(ops: "BackendOps") -> frozenset[Name]:
    """Return canonical Hydra primitive names for a backend."""
    return frozenset(bp.name for bp in ops.primitives.values())


def backend_coverage(ops: "BackendOps", required: set[str] | frozenset[str]) -> dict[str, frozenset[str]]:
    """Report supported and missing logical ops for a backend."""
    available = backend_op_names(ops)
    required = frozenset(required)
    return {
        "available": available,
        "required": required,
        "supported": frozenset(sorted(available & required)),
        "missing": frozenset(sorted(required - available)),
        "extra": frozenset(sorted(available - required)),
    }


def compare_backend_coverage(left: "BackendOps", right: "BackendOps") -> dict[str, frozenset[str]]:
    """Compare logical op coverage between two backends."""
    left_ops = backend_op_names(left)
    right_ops = backend_op_names(right)
    return {
        "shared": frozenset(sorted(left_ops & right_ops)),
        "left_only": frozenset(sorted(left_ops - right_ops)),
        "right_only": frozenset(sorted(right_ops - left_ops)),
    }


def backend_has_coverage(ops: "BackendOps", required: set[str] | frozenset[str]) -> bool:
    """Return True iff the backend supports every required logical op."""
    return frozenset(required).issubset(backend_op_names(ops))


def backend_required_for_term(required_ops: set[str] | frozenset[str], *candidates: "BackendOps") -> list["BackendOps"]:
    """Return the candidate backends which satisfy a required logical op set."""
    needed = frozenset(required_ops)
    return [ops for ops in candidates if backend_has_coverage(ops, needed)]