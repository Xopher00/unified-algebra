# Core API
from unialg.backend import Backend, NumpyApiBackend, NumpyBackend, PytorchBackend, JaxBackend, CupyBackend
from unialg.assembly.specs import PathSpec, FanSpec, FoldSpec, UnfoldSpec, LensPathSpec, LensFanSpec, FixpointSpec

# Algebra (declarations)
from unialg.algebra import Semiring, Equation, Lens, compile_einsum, semiring_contract, Sort, ProductSort
from unialg.terms import tensor_coder

# Assembly (graph construction, composition)
from unialg.assembly import assemble_graph, rebind_params
from unialg.assembly.graph import build_graph, validate_pipeline, topo_edges

# Runtime
from unialg.runtime import Program, compile_program
from unialg.runtime.program import type_check_term

# Parser
from unialg.parser import parse_ua, parse_ua_spec, UASpec

__all__ = [
    # Backends
    "Backend", "NumpyBackend", "PytorchBackend", "JaxBackend", "CupyBackend",
    # Algebra
    "Semiring", "Equation", "Lens", "Sort", "ProductSort",
    "compile_einsum", "semiring_contract", "tensor_coder",
    # Specs
    "PathSpec", "FanSpec", "FoldSpec", "UnfoldSpec",
    "LensPathSpec", "LensFanSpec", "FixpointSpec",
    # Assembly
    "assemble_graph", "rebind_params",
    # Runtime
    "Program", "compile_program",
    # Parser
    "parse_ua", "parse_ua_spec", "UASpec",
]
