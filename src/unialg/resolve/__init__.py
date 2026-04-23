# Resolution layer: compiling algebra terms into backend-callable primitives
from unialg.algebra.semiring import ResolvedSemiring
from unialg.resolve.morphism import Equation, resolve_all_primitives
from unialg.resolve.contraction import compile_einsum, semiring_contract
