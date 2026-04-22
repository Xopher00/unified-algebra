# Algebraic core: sorts, semirings, morphisms, contractions
from .semiring import semiring, resolve_semiring, ResolvedSemiring
from .sort import (
    sort, tensor_coder, sort_coder, sort_type_from_term,
    is_batched, product_sort, is_product_sort, product_sort_elements,
    check_sort_compatibility, check_rank_junction,
)
from .morphism import equation, resolve_equation, resolve_list_merge
from .contraction import compile_equation, semiring_contract
