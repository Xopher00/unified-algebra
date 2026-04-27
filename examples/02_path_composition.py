"""Path composition: chaining typed morphisms left-to-right.

Thesis: `seq` composes equations sequentially. The framework validates
sort junctions (each equation's codomain must match the next equation's
domain) and compiles the chain into a single callable.
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

seq chain : hidden -> hidden = relu >> tanh_act
''', backend)

x = np.array([-1.0, 0.0, 0.5, 2.0])

result = prog('chain', x)
expected = np.tanh(np.maximum(0, x))
np.testing.assert_allclose(result, expected, rtol=1e-6)
print(f"Input:            {x}")
print(f"relu >> tanh:     {result}")
print(f"Expected:         {expected}")
