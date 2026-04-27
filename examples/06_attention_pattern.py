"""Attention-like pattern via branch + merge chain.

Thesis: the branch merge chain (`~>`) is a stack machine that can
express multi-input patterns like scaled dot-product attention without
any attention-specific primitives.

Three branches (Q, K, V projections) feed a merge chain:
  score(Q, K) -> softmax(scores) -> mix(probs, V)

The framework sees this as: 3 parallel morphisms, then a stack-machine
reduction to 1 output. No special "attention" construct needed.
"""

import numpy as np
from unialg import parse_ua, NumpyBackend

backend = NumpyBackend()

# Template ops use nonlinearities (standing in for projections)
# so no external weights are needed at call time.
prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

# Template: same nonlinearity, different weight keys
op ~act : hidden -> hidden
  nonlinearity = abs

# Merge ops
op score : hidden -> hidden
  einsum = "ik,jk->ij"
  algebra = real

op mix : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real

# Stack trace: [Q,K,V] -> score(Q,K) -> [s,V] -> softmax(s) -> [p,V] -> mix(p,V) -> [out]
branch head : hidden -> hidden = act[q] | act[k] | act[v]
  merge = score ~> softmax ~> mix
''', backend)

x = np.array([[1.0, -1.0], [0.5, 2.0], [0.0, -0.5]], dtype=np.float32)
result = prog('head', x)

# Manual verification
q = k = v = np.abs(x)
scores = np.einsum("ik,jk->ij", q, k)
probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
probs = probs / probs.sum(axis=-1, keepdims=True)
expected = np.einsum("ij,jk->ik", probs, v)

np.testing.assert_allclose(result, expected, rtol=1e-5)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {result.shape}")
print(f"Output:\n{result}")
