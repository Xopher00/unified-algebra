# ---------------------------------------------------------------------------
# Core API — the typical user needs these
# ---------------------------------------------------------------------------
from .backend import Backend, numpy_backend, pytorch_backend
from .semiring import semiring
from .sort import sort, tensor_coder, sort_coder, is_batched
from .morphism import equation
from .graph import assemble_graph, rebind_hyperparams, build_graph
from .graph import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec

# ---------------------------------------------------------------------------
# Composition — for building paths, fans, folds, lenses manually
# ---------------------------------------------------------------------------
from .composition import path, fan
from .recursion import fold, unfold
from .lens import lens, lens_path, lens_fan

# ---------------------------------------------------------------------------
# Validation & resolution — power-user / introspection
# ---------------------------------------------------------------------------
from .morphism import resolve_equation, resolve_list_merge
from .composition import validate_path, validate_fan
from .recursion import validate_fold, validate_unfold
from .lens import validate_lens
from .graph import resolve_dag, validate_pipeline, ua_primitives, type_check_term
from .sort import sort_to_type, check_sort_junction
from .contraction import compile_equation, semiring_contract
from .semiring import resolve_semiring
