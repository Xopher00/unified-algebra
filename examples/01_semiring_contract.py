"""Semiring contraction: the same equation, different algebras.

Thesis: the semiring parameterizes *what* contraction means.
The equation "ij,j->i" is a typed morphism. Under the real semiring
it computes matrix-vector multiplication. Under the tropical semiring
it computes a shortest-path relaxation step (Bellman-Ford).

No architecture changes — only the algebra declaration differs.
"""

import numpy as np
from unialg import parse_ua, NumpyBackend

backend = NumpyBackend()

# -- Real semiring: standard matrix-vector product --

real_prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op matvec : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
''', backend)

W = np.array([[1.0, 2.0],
              [3.0, 4.0]])
x = np.array([1.0, 1.0])

result_real = real_prog('matvec', W, x)
print(f"Real semiring:     {result_real}")
# [3.0, 7.0]  — standard sum-product

# -- Tropical semiring: shortest-path relaxation --

tropical_prog = parse_ua('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec nodes(tropical)

op relax : nodes -> nodes
  einsum = "ij,j->i"
  algebra = tropical
''', backend)

# Edge weights (inf = no edge)
edges = np.array([[0.0, 1.0, np.inf],
                  [1.0, 0.0, 2.0],
                  [np.inf, 2.0, 0.0]])
dists = np.array([0.0, np.inf, np.inf])

one_hop = tropical_prog('relax', edges, dists)
print(f"Tropical (1 hop):  {one_hop}")
# [0.0, 1.0, inf] — node 1 reachable via edge 0->1

two_hop = tropical_prog('relax', edges, one_hop)
print(f"Tropical (2 hops): {two_hop}")
# [0.0, 1.0, 3.0] — node 2 reachable via 0->1->2
