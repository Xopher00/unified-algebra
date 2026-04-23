# Resolution layer: compiling algebra terms into backend-callable primitives
from unialg.resolve.ops import resolve_semiring, ResolvedSemiring
from unialg.resolve.morphism import equation, resolve_equation, resolve_list_merge, resolve_all_primitives
from unialg.resolve.contraction import compile_einsum, semiring_contract
