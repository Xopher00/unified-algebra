"""
Recursive NN — RTreeF cata.

Functor:   F(X) = Tensor + (X × X)
Semiring:  real  (add_id=0.0, mul_id=1.0)
Algebra:   leaf = W · a (contraction), node = left + right (elementwise)

No library-native counterpart — structural test only (finite output check).
"""

import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hydra.dsl.python import Left, Right

from backends import BackendSpec, TFBackend, TorchBackend, load_generated


ADD_ID = 0.0
MUL_ID = 1.0

INPUT_DIMS  = [1, 2, 3]
HIDDEN_DIMS = [1, 2, 4]

HYPO = dict(deadline=None,
            suppress_health_check=[HealthCheck.function_scoped_fixture])


BACKENDS = [
    BackendSpec(TFBackend(),    module="seed.tree", fn="fold_tree", reference=None),
    BackendSpec(TorchBackend(), module="seed.tree", fn="fold_tree", reference=None),
]


@st.composite
def _tree_recursive(draw, backend, dim, depth):
    if depth <= 0 or draw(st.booleans()):
        val = backend.random_vector(draw, dim)
        return [val], Left(val)
    else:
        left_vals,  left_tree  = draw(_tree_recursive(backend, dim, depth - 1))
        right_vals, right_tree = draw(_tree_recursive(backend, dim, depth - 1))
        return left_vals + right_vals, Right((left_tree, right_tree))


@st.composite
def tree_inputs(draw, backend, max_depth=3):
    """Draw (w, leaves, tree, input_dim, hidden_dim)."""
    input_dim  = draw(st.sampled_from(INPUT_DIMS))
    hidden_dim = draw(st.sampled_from(HIDDEN_DIMS))

    w            = backend.random_matrix(draw, hidden_dim, input_dim)
    leaves, tree = draw(_tree_recursive(backend, input_dim, max_depth))

    return w, leaves, tree, input_dim, hidden_dim


@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param


class TestTreeRnn:

    @given(data=st.data())
    @settings(max_examples=50, **HYPO)
    def test_tree_runs(self, spec, data):
        backend = spec.backend
        w, _, tree, _, _ = data.draw(tree_inputs(backend))

        fold   = load_generated(spec.module, spec.fn)
        result = fold(w, tree)

        assert backend.is_finite(result), \
            f"[{backend}] non-finite tree output"
