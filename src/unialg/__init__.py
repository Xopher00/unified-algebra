# Core API
from unialg.backend import Backend, NumpyApiBackend, NumpyBackend, PytorchBackend, JaxBackend, CupyBackend
from unialg.assembly.specs import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec, LensFanSpec, FixpointSpec

# Algebra (declarations)
from unialg.algebra import (
    Semiring,
    Equation, Lens,
    compile_einsum, semiring_contract,
    Sort, ProductSort, tensor_coder,
)

# Assembly (compilation, composition, validation, graph)
from unialg.assembly import (
    assemble_graph, rebind_hyperparams, build_graph,
    validate_pipeline, topo_edges,
    resolve_equation, resolve_equation_as_merge,
    path, fan, fold, unfold, fixpoint, lens_path, lens_fan,
)

# Runtime
from unialg.runtime import Program, compile_program, type_check_term

# Parser
from unialg.parser import parse_ua, parse_ua_spec, UASpec
