"""Batched sorts and named axes.

Thesis: sorts carry optional axis names and batch dimensions.
The framework auto-prepends batch dimensions to einsum subscripts
and validates axis compatibility at composition boundaries.

Batching is declared on the sort, not on the equation. When a sort
is `batched`, every equation using it operates independently on
each sample in the batch.
"""

import numpy as np
from unialg import parse_ua, NumpyBackend

backend = NumpyBackend()

prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real, batched, axes=[feature])

op relu : hidden -> hidden
  nonlinearity = relu

op tanh_act : hidden -> hidden
  nonlinearity = tanh

seq chain : hidden -> hidden = relu >> tanh_act
''', backend)

# Batch of 3 vectors, each of dimension 4
X = np.array([[-1.0, 0.5, 2.0, -0.3],
              [ 0.0, 1.0, -1.0, 0.7],
              [ 3.0, -2.0, 0.1, 0.0]])

result = prog('chain', X)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {result.shape}")
assert result.shape == (3, 4)

expected = np.tanh(np.maximum(0, X))
np.testing.assert_allclose(result, expected, rtol=1e-6)
print("Batch dimension preserved correctly.")
print(f"Output:\n{result}")
