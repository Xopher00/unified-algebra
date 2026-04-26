# Assembly layer: graph construction, validation, and composition
from unialg.assembly.graph import assemble_graph, rebind_hyperparams, build_graph, validate_pipeline, topo_edges
from unialg.assembly._primitives import (
    unfold_n_primitive, fixpoint_primitive,
    lens_fwd_primitive, lens_bwd_primitive, residual_add_primitive,
)
