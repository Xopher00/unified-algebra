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
          "result_type": "FLOAT"
        }
      }
    }

``arg_type`` and ``result_type`` are Hydra type declarations parsed by
``type_from_spec``; TermCoders are derived automatically via ``coder_for_type``.
Supported shorthands: ``"FLOAT"``, ``"INT"``, ``"STRING"``, ``"BOOL"``,
``"BINARY"``, ``"UNIT"``.  Structured types (``{"list": T}``, ``{"pair": [A, B]}``,
etc.) are also valid for backend ops that accept/return lists, pairs, or
optional values in universal Python representation.
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hydra.context import Context
from hydra.core import LiteralType, Name, Type, TypeLiteral
from hydra.graph import Graph, Primitive, TermCoder
from hydra.lib import maps as Maps
from hydra.packaging import Library, Namespace
import hydra.dsl.meta.phantoms as P

from unialg.objects import ExpType, TypeScheme, ProductType
from unialg.semantics import morphisms
from unialg.semantics.typeops import _EMPTY_GRAPH
from unialg.structure import terms as struct_terms
from unialg.syntax import expressions as expr

from .codecs import type_from_spec, coder_for_type, _expect_right
from .runtime_store import RuntimeStore

_CANONICAL_PREFIX = "unialg.backend"


def _is_binary_type(typ: Type) -> bool:
    return isinstance(typ, TypeLiteral) and typ.value == LiteralType.BINARY


@dataclass(frozen=True)
class BinaryAdapter:
    """Generic boundary adapter: bytes ↔ native tensor, using BytesIO framing.

    Resolved from the ``"binary_adapter"`` dict in a backend JSON spec.
    Knows about calling conventions (``dump_style``) but not framework names.

    ``dump_style`` values:
      ``"file_first"``  — ``dump_fn(buf, value)``  e.g. ``numpy.save``, ``cupy.save``
      ``"value_first"`` — ``dump_fn(value, buf)``  e.g. ``torch.save``

    Unknown ``dump_style`` values raise ``ValueError`` at construction — no silent fallback.
    ``load_kwargs`` are forwarded verbatim to ``load_fn`` (e.g. ``weights_only=False`` for torch).
    """

    dump_fn: Callable
    load_fn: Callable
    dump_style: str
    load_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dump_style not in ("file_first", "value_first"):
            raise ValueError(
                f"BinaryAdapter: unknown dump_style {self.dump_style!r}. "
                "Expected 'file_first' or 'value_first'."
            )

    def dump(self, native) -> bytes:
        buf = io.BytesIO()
        if self.dump_style == "file_first":
            self.dump_fn(buf, native)
        elif self.dump_style == "value_first":
            self.dump_fn(native, buf)
        return buf.getvalue()

    def load(self, b: bytes):
        return self.load_fn(io.BytesIO(b), **self.load_kwargs)


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
    binary_adapter: object | None = None
    store: RuntimeStore | None = None

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



def register_backend_primitive(
    canonical_name: str,
    path: str | Callable,
    arg_type: Type,
    arity: int,
    *,
    arg_coder: TermCoder,
    result_coder: TermCoder,
    result_type: Type | None = None,
    binary_adapter=None,
    store: RuntimeStore | None = None,
) -> BackendPrimitive:
    """Register a single backend function as a Hydra primitive.

    Args:
        canonical_name: Hydra primitive name, e.g. ``"unialg.backend.add"``.
            This name is the same across all backends for the same logical op.
        path: Dotted import path to the backend function, e.g. ``"numpy.add"``.
        arg_type: Hydra ``Type`` for all input arguments (homogeneous).
        arity: Number of arguments the function takes.  Must be provided
            explicitly — do not rely on ``infer_arity`` for arbitrary APIs.
        arg_coder: TermCoder used to decode arguments before calling the function.
        result_coder: TermCoder used to encode the Python result back to a Hydra term.
        result_type: Hydra ``Type`` for the return value.  Defaults to
            ``arg_type`` when omitted.

    Returns:
        A ``BackendPrimitive`` whose ``primitive`` field is registered under
        ``canonical_name`` with the appropriate curried ``TypeScheme``.
    """
    fn = resolve_function(path) if isinstance(path, str) else path
    result_type = result_type or arg_type
    name = Name(canonical_name)
    scheme = TypeScheme((), _curried_type(arg_type, result_type, arity), None)

    use_store = (
        binary_adapter is not None
        and store is not None
        and _is_binary_type(arg_type)
        and _is_binary_type(result_type)
    )

    def impl(ctx: Context, graph: Graph, args):
        """Decode Hydra arguments, call the backend function, and re-encode."""
        py_args = [
            _expect_right(
                arg_coder.encode(ctx, graph, arg),
                f"decoding argument for {canonical_name}",
            )
            for arg in args
        ]
        if use_store:
            native_args = [store.get(key) for key in py_args]
            native_result = fn(*native_args)
            result_key = store.put(native_result)
            return result_coder.decode(ctx, result_key)
        py_result = fn(*py_args)
        return result_coder.decode(ctx, py_result)

    return BackendPrimitive(
        primitive=Primitive(name, scheme, impl),
        arity=arity,
        arg_type=arg_type,
        result_type=result_type,
        arg_coder=arg_coder,
        result_coder=result_coder,
        binary_adapter=binary_adapter,
        store=store,
    )


def _primitive_morphism(bp: BackendPrimitive) -> morphisms.Morphism:
    """Wrap a ``BackendPrimitive`` as a ``Morphism``.

    Builds a lambda over the product domain that applies the primitive in
    curried form, carrying the primitive in ``aux_primitives`` so ``run``
    can register it in a temporary graph automatically.
    """
    x = P.var("x")
    args = struct_terms.product_arg(x, bp.arity)
    term = P.primitive(bp.name)
    for arg in args:
        term = P.apply(term, arg)
    raw = struct_terms.normalize_term(P.lam("x", term)).value
    dom = repeated_product(bp.arg_type, bp.arity)
    return morphisms.Morphism(
        expr.Prim(raw, dom, bp.result_type),
        aux_primitives=(bp.primitive,),
    )


def load_spec(
    spec: dict | str | Path,
) -> tuple[dict[str, BackendPrimitive], object | None, RuntimeStore | None]:
    """Load a backend spec and return primitives, the binary adapter module, and the store.

    ``spec`` may be a parsed dict, a file path, or a path string.  Each entry
    in ``spec["operations"]`` becomes one ``BackendPrimitive`` keyed by its
    logical op name (e.g. ``"add"``, ``"reduce.add"``).

    If the spec declares a top-level ``"binary_adapter"`` dotted module path, that module
    is imported and a ``RuntimeStore`` is created.  All BINARY→BINARY primitives in the
    spec will use the store path; others fall through to the normal coder path.

    The canonical Hydra name is derived as ``unialg.backend.<op_name>``.
    """
    if isinstance(spec, (str, Path)):
        with open(spec) as f:
            spec = json.load(f)

    adapter_spec = spec.get("binary_adapter")
    if adapter_spec is None:
        binary_adapter = None
    elif isinstance(adapter_spec, str):
        binary_adapter = importlib.import_module(adapter_spec)
    elif isinstance(adapter_spec, dict):
        binary_adapter = BinaryAdapter(
            dump_fn=resolve_function(adapter_spec["dump"]),
            load_fn=resolve_function(adapter_spec["load"]),
            dump_style=adapter_spec.get("dump_style", "file_first"),
            load_kwargs=adapter_spec.get("load_kwargs") or {},
        )
    else:
        raise ValueError(f"binary_adapter must be a string or dict, got {type(adapter_spec)!r}")
    store: RuntimeStore | None = RuntimeStore() if binary_adapter is not None else None

    result: dict[str, BackendPrimitive] = {}
    for op_name, entry in spec["operations"].items():
        arg_type    = type_from_spec(entry["arg_type"])
        result_type = type_from_spec(entry.get("result_type", entry["arg_type"]))
        arg_coder   = coder_for_type(arg_type)
        result_coder = coder_for_type(result_type)
        canonical = f"{_CANONICAL_PREFIX}.{op_name}"
        result[op_name] = register_backend_primitive(
            canonical,
            entry["path"],
            arg_type,
            entry["arity"],
            arg_coder=arg_coder,
            result_coder=result_coder,
            result_type=result_type,
            binary_adapter=binary_adapter,
            store=store,
        )
    return result, binary_adapter, store


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

    def __init__(
        self,
        primitives: dict[str, BackendPrimitive],
        binary_adapter=None,
        store: RuntimeStore | None = None,
    ):
        self._primitives = primitives
        self._binary_adapter = binary_adapter
        self._store = store
        self._library = backend_library(primitives)
        self._morphisms: dict[str, morphisms.Morphism] = {
            name: _primitive_morphism(bp) for name, bp in primitives.items()
        }

    @classmethod
    def from_spec(cls, spec: dict | str | Path) -> "BackendOps":
        """Construct from a JSON spec file path, path string, or parsed dict."""
        primitives, binary_adapter, store = load_spec(spec)
        return cls(primitives, binary_adapter=binary_adapter, store=store)

    @property
    def library(self) -> Library:
        """The Hydra-native primitive library for this backend."""
        return self._library

    @property
    def primitives(self) -> dict[str, BackendPrimitive]:
        """The registered backend primitives keyed by logical op name."""
        return self._primitives

    @property
    def binary_adapter(self):
        """The backend binary adapter module (with load/dump), or None."""
        return self._binary_adapter

    @property
    def store(self) -> RuntimeStore | None:
        """The RuntimeStore for native tensor lifetime, or None."""
        return self._store

    def put_input(self, v) -> bytes:
        """Convert an external input to native and return its store handle.

        If ``v`` is bytes, uses ``binary_adapter.load`` to deserialize.
        Otherwise stores the value as-is (caller has already produced a native value).
        Returns a 16-byte UUID handle for encoding as a BINARY Term.
        """
        native = self._binary_adapter.load(v) if isinstance(v, bytes) else v
        return self._store.put(native)

    def get_output(self, key: bytes):
        """Retrieve the final native result from the store by handle bytes."""
        return self._store.get(key)

    def register(self, name: str, bp: BackendPrimitive) -> None:
        """Register a custom BackendPrimitive under logical op name ``name``.

        After registration, ``ops[name]`` returns the corresponding Morphism,
        identically to built-in ops loaded from a JSON spec.  Callers
        (structure/semiring_factory.py) use this to register custom callables
        before constructing a Semiring.
        """
        self._primitives[name] = bp
        m = _primitive_morphism(bp)
        self._morphisms[name] = m
        # Re-build the library so to_graph() picks up the new primitive.
        self._library = backend_library(self._primitives)

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
    return Maps.from_list(list({prim.name: prim for prim in library.primitives}.items()))


def library_to_graph(library: Library, base: Graph | None = None) -> Graph:
    """Install a backend Library's primitives into a Hydra Graph."""
    base = base or _EMPTY_GRAPH

    prims = dict(base.primitives)
    for prim in library.primitives:
        prims[prim.name] = prim

    return Graph(
        bound_terms=base.bound_terms,
        bound_types=base.bound_types,
        class_constraints=base.class_constraints,
        lambda_variables=base.lambda_variables,
        metadata=base.metadata,
        primitives=Maps.from_list(list(prims.items())),
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
