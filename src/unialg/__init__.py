# Core API
from unialg.backend import Backend, numpy_backend, pytorch_backend
from unialg.specs import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec, LensFanSpec, FixpointSpec

# Algebra (term construction)
from unialg.algebra import (
    semiring,
    sort, tensor_coder, sort_coder, sort_type_from_term,
    is_batched, product_sort, is_product_sort,
    fixpoint,
)

# Resolve (runtime compilation)
from unialg.resolve import (
    resolve_semiring,
    Equation, resolve_all_primitives,
    compile_einsum, semiring_contract,
)

# Composition
from unialg.composition import path, fan, fold, unfold, lens, lens_path, validate_lens

# Assembly
from unialg.assembly import (
    assemble_graph, rebind_hyperparams, build_graph,
    validate_spec, validate_pipeline, topo_edges,
)

# Runtime
from unialg.runtime import Program, compile_program, type_check_term

# Parser
from unialg.parser import parse_ua, parse_ua_spec, UASpec
