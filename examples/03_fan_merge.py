"""Fan (branch) composition: parallel branches merged by a stack machine.

Thesis: `branch` applies multiple morphisms to the same input, then
merges their outputs via a stack-machine chain. This expresses
multi-head patterns without special-casing any architecture.

The merge chain `~>` operates on a stack: each step consumes operands
from the top, pushes its result, and unconsumed values carry through.
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

# Elementwise product of two branches
op hadamard : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch gated : hidden -> hidden = relu | tanh_act
  merge = hadamard
''', backend)

x = np.array([-1.0, 0.5, 2.0])
result = prog('gated', x)
print(f"relu(x) * tanh(x): {result}")
# relu([-1, 0.5, 2]) * tanh([-1, 0.5, 2])
# = [0, 0.5, 2] * [-0.762, 0.462, 0.964]
# = [0.0, 0.231, 1.928]
