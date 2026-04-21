# ---------------------------------------------------------------------------
# Core API — the typical user needs these
# ---------------------------------------------------------------------------
from .program import Program, compile_program
from .backend import Backend, numpy_backend, pytorch_backend
from .semiring import semiring
from .sort import sort, tensor_coder, sort_coder, is_batched, product_sort, is_product_sort
from .morphism import equation
from .graph import assemble_graph, rebind_hyperparams, build_graph
from .specs import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec, FixpointSpec

# ---------------------------------------------------------------------------
# Composition — for building paths, fans, folds, lenses manually
# ---------------------------------------------------------------------------
from .composition import path, fan
from .recursion import fold, unfold
from .composition import lens, lens_path
from .recursion import fixpoint

# ---------------------------------------------------------------------------
# Validation & resolution — power-user / introspection
# ---------------------------------------------------------------------------
from .morphism import resolve_equation, resolve_list_merge
from .validation import validate_spec
from .composition import validate_lens
from .validation import resolve_dag, validate_pipeline
from .graph import type_check_term
from .contraction import compile_equation, semiring_contract
from .semiring import resolve_semiring

# ---------------------------------------------------------------------------
# Parser — Phase 15
# ---------------------------------------------------------------------------
from .parser import parse_ua, parse_ua_spec, UASpec
