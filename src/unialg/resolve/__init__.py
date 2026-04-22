# Resolution layer: compiling algebra terms into backend-callable primitives
from unialg.resolve.ops import resolve_semiring, ResolvedSemiring
from unialg.assembly.primitives import (
    unfold_n_primitive, fixpoint_primitive, lens_fwd_primitive, lens_bwd_primitive,
)
from unialg.resolve.morphism import equation, resolve_equation, resolve_list_merge, resolve_all_primitives
from unialg.resolve.contraction import compile_einsum, semiring_contract
