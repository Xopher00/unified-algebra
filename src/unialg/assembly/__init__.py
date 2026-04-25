# Assembly layer: graph construction, validation, resolution, and composition
from unialg.assembly.graph import assemble_graph, rebind_hyperparams, build_graph
from unialg.assembly.pipeline import validate_pipeline, topo_edges, EquationPipeline
from unialg.assembly.resolver import resolve_equation, resolve_equation_as_merge
from unialg.assembly._composition import path, fan, fold, unfold, fixpoint, lens_path, lens_fan
from unialg.assembly._primitives import (
    unfold_n_primitive, fixpoint_primitive,
    lens_fwd_primitive, lens_bwd_primitive, residual_add_primitive,
)
