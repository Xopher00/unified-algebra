"""Backend primitive registration for the unialg DSL.

Translates backend JSON specs into Hydra primitives. Semantic `Morphism`
wrapping happens in `unialg.main` when a backend is loaded for a source program.

Pipeline::

    JSON spec
      -> load_spec             (resolve paths, look up types and codecs)
      -> register_backend_primitive  (build Hydra Primitive with typed impl)
      -> BackendOps            (hold primitives, store, and native boundary)

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

``arg_type`` or ``arg_types`` and ``result_type`` are Hydra type declarations parsed by
``type_from_spec``; TermCoders are derived automatically via ``coder_for_type``.
Supported shorthands: ``"FLOAT"``, ``"INT"``, ``"STRING"``, ``"BOOL"``,
``"BINARY"``, ``"UNIT"``.  Structured types (``{"list": T}``, ``{"pair": [A, B]}``,
etc.) are also valid for backend ops that accept/return lists, pairs, or
optional values in universal Python representation. The ``kind`` field is
descriptive metadata; loading semantics are determined by ``path``, ``arity``,
``arg_type``/``arg_types``, and ``result_type``.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from hydra.context import Context
from hydra.core import Name, Type
from hydra.dsl.python import Nothing
from hydra.graph import Graph, Primitive, TermCoder
from hydra.lib import maps as Maps
from hydra.packaging import Library, Namespace

from unialg.objects import ExpType, ProductType, TypeScheme, standard_graph

from .codecs import type_from_spec, coder_for_type, expect_right
from .boundary import (
    BinaryAdapter,
    RuntimeStore,
    decode_boundary_output,
    encode_boundary_input,
    is_binary_type,
)

_CANONICAL_PREFIX = "unialg.backend"


@dataclass(frozen=True)
class BackendPrimitive:
    """A resolved backend operation ready for Hydra primitive registration.

    Carries the Hydra ``Primitive`` (name + type scheme + impl), the arity,
    the Hydra types for arguments and result, and the pre-computed domain type.
    """

    primitive: Primitive
    arity: int
    arg_types: tuple[Type, ...]
    result_type: Type
    dom: Type
    arg_coders: tuple[TermCoder, ...]
    result_coder: TermCoder
    fn: Callable
    # Operation metadata. BackendOps owns the adapter used at the whole-program
    # I/O boundary; primitive impls only need the RuntimeStore.
    binary_adapter: object | None = None
    store: RuntimeStore | None = None

    @property
    def name(self) -> Name:
        """Canonical Hydra name, e.g. ``Name("unialg.backend.add")``."""
        return self.primitive.name

    @property
    def arg_type(self) -> Type:
        """First argument type, retained for single-carrier consumers."""
        return self.arg_types[0]

    @property
    def arg_coder(self) -> TermCoder:
        """First argument coder, retained for tensor carrier consumers."""
        return self.arg_coders[0]


def resolve_function(path: str):
    """Import and return a callable by dotted path.

    The last segment is the attribute name; everything before it is the module::

        resolve_function("numpy.add")           -> numpy.add
        resolve_function("jax.numpy.add")       -> jax.numpy.add
        resolve_function("scipy.special.expit") -> scipy.special.expit
    """
    module_name, function_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), function_name)


def _resolve_operation_function(backend_name: str, op_name: str, path: str):
    """Resolve one backend operation path with spec context in the error."""
    try:
        return resolve_function(path)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(
            f"backend {backend_name!r} op {op_name!r}: "
            f"cannot resolve path {path!r}: {e}"
        ) from e


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


def _curried_type(arg_types: tuple[Type, ...], result_type: Type):
    """Build the curried Hydra type for the declared argument sequence."""
    typ = result_type
    for arg_type in reversed(arg_types):
        typ = ExpType(arg_type, typ)
    return typ


def _argument_domain(arg_types: tuple[Type, ...]) -> Type:
    """Build the left-nested product shape used for multi-argument execution."""
    dom = arg_types[0]
    for arg_type in arg_types[1:]:
        dom = ProductType(dom, arg_type)
    return dom


def register_backend_primitive(
    canonical_name: str, path: str | Callable,
    arg_type: Type | None, arity: int, *,
    arg_coder: TermCoder | None, result_coder: TermCoder,
    result_type: Type | None = None,
    arg_types: tuple[Type, ...] | None = None,
    arg_coders: tuple[TermCoder, ...] | None = None,
    binary_adapter=None,
    store: RuntimeStore | None = None,
) -> BackendPrimitive:
    """Register a single backend function as a Hydra primitive.

    Args:
        canonical_name: Hydra primitive name, e.g. ``"unialg.backend.add"``.
            This name is the same across all backends for the same logical op.
        path: Dotted import path to the backend function, e.g. ``"numpy.add"``.
        arg_type: Legacy homogeneous Hydra input type.
        arity: Number of arguments the function takes.  Must be provided
            explicitly — do not rely on ``infer_arity`` for arbitrary APIs.
        arg_coder: Legacy homogeneous input coder.
        result_coder: TermCoder used to encode the Python result back to a Hydra term.
        result_type: Hydra ``Type`` for the return value.  Defaults to the
            first declared argument type when omitted.

    Returns:
        A ``BackendPrimitive`` whose ``primitive`` field is registered under
        ``canonical_name`` with the appropriate curried ``TypeScheme``.
    """
    fn = resolve_function(path) if isinstance(path, str) else path
    arg_types = arg_types or ((arg_type,) * arity if arg_type is not None else ())
    if not arg_types:
        raise ValueError(f"{canonical_name}: arg_types or arg_type must be provided")
    if len(arg_types) != arity:
        raise ValueError(f"{canonical_name}: arity {arity} does not match {len(arg_types)} argument types")
    arg_coders = arg_coders or ((arg_coder,) * arity if arg_coder is not None else ())
    if len(arg_coders) != arity:
        raise ValueError(f"{canonical_name}: arity {arity} does not match {len(arg_coders)} argument coders")
    result_type = result_type or arg_types[0]
    name = Name(canonical_name)
    scheme = TypeScheme((), _curried_type(arg_types, result_type), Nothing())

    store_args = tuple(store is not None and is_binary_type(t) for t in arg_types)
    store_result = store is not None and is_binary_type(result_type)

    def impl(ctx: Context, graph: Graph, args, *, fn=fn, 
        arg_coders=arg_coders, result_coder=result_coder,
        canonical_name=canonical_name,
        store=store, store_args=store_args, 
        store_result=store_result,
    ):
        """Decode Hydra arguments, call the backend function, and re-encode."""
        py_args = [
            expect_right(
                arg_coder.encode(ctx, graph, arg),
                f"decoding argument for {canonical_name}",
            )
            for arg_coder, arg in zip(arg_coders, args)
        ]
        py_args = [
            store.get(value) if use_store else value
            for value, use_store in zip(py_args, store_args)
        ]
        py_result = fn(*py_args)
        if store_result:
            py_result = store.put(py_result)
        return result_coder.decode(ctx, py_result)

    return BackendPrimitive(
        primitive=Primitive(name, scheme, impl),
        arity=arity,
        arg_types=arg_types,
        result_type=result_type,
        dom=_argument_domain(arg_types),
        arg_coders=arg_coders,
        result_coder=result_coder,
        fn=fn,
        binary_adapter=binary_adapter,
        store=store,
    )


def _resolve_structural(spec: dict) -> dict[str, Callable]:
    """Resolve non-user-facing backend structural callables from spec metadata."""
    backend_name = spec.get("backend", "<unknown>")
    return {
        name: _resolve_operation_function(backend_name, f"structural.{name}", path)
        for name, path in spec.get("structural", {}).items()
    }


def _resolve_binary_adapter(adapter_spec) -> object | None:
    if adapter_spec is None:
        return None
    if isinstance(adapter_spec, str):
        return importlib.import_module(adapter_spec)
    if isinstance(adapter_spec, dict):
        return BinaryAdapter(
            dump_fn=resolve_function(adapter_spec["dump"]),
            load_fn=resolve_function(adapter_spec["load"]),
            dump_style=adapter_spec.get("dump_style", "file_first"),
            load_kwargs=adapter_spec.get("load_kwargs") or {},
        )
    raise ValueError(f"binary_adapter must be a string or dict, got {type(adapter_spec)!r}")


def _prepare_op(backend_name: str, op_name: str, entry: dict) -> tuple:
    arg_specs    = entry.get("arg_types")
    if arg_specs is None:
        arg_specs = [entry["arg_type"]] * entry["arity"]
    if len(arg_specs) != entry["arity"]:
        raise ValueError(f"backend {backend_name!r} op {op_name!r}: arity does not match arg_types")
    arg_types    = tuple(type_from_spec(spec) for spec in arg_specs)
    result_type  = type_from_spec(entry.get("result_type", arg_specs[0]))
    arg_coders   = tuple(coder_for_type(arg_type) for arg_type in arg_types)
    result_coder = coder_for_type(result_type)
    canonical    = f"{_CANONICAL_PREFIX}.{op_name}"
    fn           = _resolve_operation_function(backend_name, op_name, entry["path"])
    return (op_name, entry, arg_types, result_type, arg_coders, result_coder, canonical, fn)


def load_spec(
    spec: dict | str | Path,
) -> tuple[dict[str, BackendPrimitive], object | None, RuntimeStore | None]:
    """Load a backend spec and return primitives, binary adapter, and store.

    ``spec`` may be a parsed dict, a file path, or a path string.  Each entry
    in ``spec["operations"]`` becomes one ``BackendPrimitive`` keyed by its
    logical op name (e.g. ``"add"``, ``"reduce.add"``).

    If any operation uses the BINARY handle type, a ``RuntimeStore`` is
    created. A top-level ``"binary_adapter"`` is only needed for serialized
    byte inputs/outputs; native values can be stored directly.

    The canonical Hydra name is derived as ``unialg.backend.<op_name>``.
    """
    if isinstance(spec, (str, Path)):
        with open(spec) as f:
            spec = json.load(f)

    backend_name = spec.get("backend", "<unknown>")
    binary_adapter = _resolve_binary_adapter(spec.get("binary_adapter"))
    prepared = [_prepare_op(backend_name, op_name, entry)
                for op_name, entry in spec["operations"].items()]

    needs_store = any(
        any(is_binary_type(arg_type) for arg_type in arg_types) or is_binary_type(result_type)
        for _, _, arg_types, result_type, *_ in prepared
    )
    store: RuntimeStore | None = RuntimeStore() if needs_store else None

    result: dict[str, BackendPrimitive] = {}
    for op_name, entry, arg_types, result_type, arg_coders, result_coder, canonical, fn in prepared:
        result[op_name] = register_backend_primitive(
            canonical, fn, None, entry["arity"],
            arg_coder=None,
            arg_types=arg_types,
            arg_coders=arg_coders,
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
    """All runtime primitives for a backend, keyed by logical op name.

    Canonical Hydra names follow ``unialg.backend.<op>``
    (e.g. ``unialg.backend.add``, ``unialg.backend.reduce.add``).
    Access primitives by their short spec key: ``ops["add"]``,
    ``ops["reduce.add"]``.

    BackendOps also owns the RuntimeStore and binary adapter used by whole-program
    I/O.  The orchestration layer wraps these primitives as semantic morphisms.

    Example::

        ops = BackendOps.from_spec("backends/numpy.json")
        add = ops["add"]   # BackendPrimitive: FLOAT × FLOAT → FLOAT
    """

    def __init__(
        self,
        primitives: dict[str, BackendPrimitive],
        binary_adapter=None,
        store: RuntimeStore | None = None,
        structural: dict[str, Callable] | None = None,
    ):
        self._primitives = primitives
        self._binary_adapter = binary_adapter
        self._store = store
        self._structural = dict(structural or {})
        self._library = backend_library(primitives)

    @classmethod
    def from_spec(cls, spec: dict | str | Path) -> "BackendOps":
        """Construct from a JSON spec file path, path string, or parsed dict."""
        if isinstance(spec, (str, Path)):
            with open(spec) as f:
                spec_dict = json.load(f)
        else:
            spec_dict = spec
        structural = _resolve_structural(spec_dict)
        primitives, binary_adapter, store = load_spec(spec_dict)
        return cls(primitives, binary_adapter=binary_adapter, store=store, structural=structural)

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

    @property
    def structural(self) -> dict[str, Callable]:
        """Non-user-facing backend structural callables from backend JSON metadata."""
        return self._structural

    def structural_op(self, name: str) -> Callable:
        """Return a structural callable declared in backend JSON metadata."""
        return self._structural[name]

    def _put_input(self, v) -> bytes:
        """Convert an external input to native and return its store handle.

        If ``v`` is bytes, uses ``binary_adapter.load`` to deserialize.
        Otherwise stores the value as-is (caller has already produced a native value).
        Returns a 16-byte UUID handle for encoding as a BINARY Term.
        """
        native = self._binary_adapter.load(v) if isinstance(v, bytes) else v
        return self._store.put(native)

    def _get_output(self, key: bytes):
        """Retrieve the final native result from the store by handle bytes."""
        return self._store.get(key)

    def encode_boundary_input(self, typ: Type, value):
        """Encode a whole-program input at the backend/native boundary."""
        return encode_boundary_input(typ, value, self._put_input)

    def decode_boundary_output(self, typ: Type, value):
        """Decode a whole-program output at the backend/native boundary."""
        return decode_boundary_output(typ, value, self._get_output)

    def register(self, name: str, bp: BackendPrimitive) -> None:
        """Register a custom BackendPrimitive under logical op name ``name``."""
        self._primitives[name] = bp
        self._library = backend_library(self._primitives)

    def __getitem__(self, name: str) -> BackendPrimitive:
        """Return the BackendPrimitive for logical op ``name``."""
        return self._primitives[name]

    def __contains__(self, name: str) -> bool:
        return name in self._primitives

    def keys(self):
        """Return all registered logical op names."""
        return self._primitives.keys()

    @property
    def op_names(self) -> frozenset[str]:
        """Logical operation names supported by this backend."""
        return frozenset(self._primitives.keys())

    @property
    def hydra_names(self) -> frozenset[Name]:
        """Canonical Hydra primitive names supported by this backend."""
        return frozenset(bp.name for bp in self._primitives.values())

    def to_graph(self, base: Graph | None = None) -> Graph:
        """Install this backend's primitives into a Hydra Graph."""
        return library_to_graph(self._library, base if base is not None else standard_graph())

    @staticmethod
    def build_graph(backends: "dict[str, BackendOps]") -> Graph:
        """Build a Hydra graph from a collection of backends, falling back to standard_graph() when empty."""
        g = standard_graph()
        for ops in backends.values():
            g = ops.to_graph(g)
        return g

    def coverage(self, required: set[str] | frozenset[str]) -> dict[str, frozenset[str]]:
        """Report supported and missing logical ops for this backend."""
        available = self.op_names
        required = frozenset(required)
        return {
            "available": available,
            "required": required,
            "supported": frozenset(sorted(available & required)),
            "missing": frozenset(sorted(required - available)),
            "extra": frozenset(sorted(available - required)),
        }


def library_primitives_map(library: Library):
    """Convert a Hydra Library into a Graph.primitives-style map."""
    return Maps.from_list(list({prim.name: prim for prim in library.primitives}.items()))


def library_to_graph(library: Library, base: Graph | None = None) -> Graph:
    """Install a backend Library's primitives into a Hydra Graph."""
    base = base or standard_graph()

    prims = dict(base.primitives)
    for prim in library.primitives:
        prims[prim.name] = prim

    return replace(base, primitives=Maps.from_list(list(prims.items())))


