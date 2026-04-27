"""Custom semiring via `define`: user-defined operations inline.

Thesis: users can define new operations in the DSL without Python code.
A `define` compiles an expression over existing backend ops into a
callable, then registers it for use in algebra declarations.

Here we build a log-space semiring where plus = logaddexp,
demonstrating that the framework is truly operation-agnostic.
"""

import numpy as np
from unialg import parse_ua, NumpyBackend

backend = NumpyBackend()

# -- Smooth tropical: temperature-controlled interpolation --
# At T=1 this is logaddexp; at T->0 it approaches hard max (tropical).

T = 0.5
prog = parse_ua(f'''
define binary smooth_max(a, b) = {T} * logaddexp(a / {T}, b / {T})

algebra smooth_trop(plus=smooth_max, times=add, zero=-inf, one=0.0)
spec hidden(smooth_trop)

op relax : hidden -> hidden
  einsum = "ij,j->i"
  algebra = smooth_trop
''', backend)

W = np.array([[0.0, -1.0],
              [1.0,  0.0]])
x = np.array([2.0, 3.0])

result = prog('relax', W, x)

# Verify against manual computation
expected = np.array([
    T * np.logaddexp((0.0 + 2.0) / T, (-1.0 + 3.0) / T),
    T * np.logaddexp((1.0 + 2.0) / T, ( 0.0 + 3.0) / T),
])
np.testing.assert_allclose(result, expected, rtol=1e-6)
print(f"Smooth tropical (T={T}): {result}")
print(f"Expected:                {expected}")
