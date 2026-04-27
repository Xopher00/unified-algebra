"""Residual (skip) connection via seq+.

Thesis: appending `+` to a `seq` name adds a skip connection:
output = seq(x) + x, where `+` is the semiring's `plus` operation.

Under the real semiring, this is additive skip (ResNet-style).
Under a different semiring, it would be the corresponding algebraic
operation — the skip connection is parameterized by the algebra.
"""

import numpy as np
from unialg import parse_ua, NumpyBackend

backend = NumpyBackend()

prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

op tanh_act : hidden -> hidden
  nonlinearity = tanh

# The + suffix adds: output = (relu >> tanh)(x) + x
seq resblock+ : hidden -> hidden = relu >> tanh_act
  algebra = real
''', backend)

x = np.array([-1.0, 0.5, 2.0])

result = prog('resblock', x)

# Manual: tanh(relu(x)) + x
transformed = np.tanh(np.maximum(0, x))
expected = transformed + x
np.testing.assert_allclose(result, expected, rtol=1e-6)
print(f"Input:         {x}")
print(f"Transformed:   {transformed}")
print(f"With residual: {result}")
print(f"Expected:      {expected}")
