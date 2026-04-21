# ---------------------------------------------------------------------------
# Core API — the typical user needs these
# ---------------------------------------------------------------------------
from .backend import Backend, numpy_backend, pytorch_backend
from .semiring import semiring
from .sort import sort, tensor_coder, sort_coder, is_batched, product_sort, is_product_sort
from .morphism import equation
from .graph import assemble_graph, rebind_hyperparams, build_graph
from .graph import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec, FixpointSpec

# ---------------------------------------------------------------------------
# Composition — for building paths, fans, folds, lenses manually
# ---------------------------------------------------------------------------
from .composition import path, fan
from .recursion import fold, unfold
from .composition import lens, lens_path, lens_fan
from .recursion import fixpoint

# ---------------------------------------------------------------------------
# Validation & resolution — power-user / introspection
# ---------------------------------------------------------------------------
from .morphism import resolve_equation, resolve_list_merge
from .validation import validate_spec
from .composition import validate_lens
from .graph import resolve_dag, validate_pipeline, ua_primitives, type_check_term
from .sort import sort_to_type, check_sort_junction
from .contraction import compile_equation, semiring_contract
from .semiring import resolve_semiring
