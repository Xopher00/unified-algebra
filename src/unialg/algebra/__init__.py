# Algebraic core: sorts, semirings, equations, lenses, contraction
from unialg.algebra.semiring import Semiring
from unialg.algebra.equation import Equation
from unialg.algebra.lens import Lens
from unialg.algebra.contraction import compile_einsum, semiring_contract
from unialg.algebra.sort import Sort, ProductSort, sort_wrap, check_sort_compatibility
from unialg.terms import tensor_coder
