"""
Backend-parameterized Hypothesis strategies.

Tensors are generated in the target backend's native format.
No runtime crossing.
"""

from hypothesis import strategies as st
from hydra.dsl.python import Left, Right

from backends import SCALAR, VECTOR, MATRIX, VECTOR_DIMS, MATRIX_DIMS


_floats = st.floats(min_value=-2, max_value=2,
                    allow_nan=False, allow_infinity=False)

INPUT_DIMS = [1, 2, 3]
HIDDEN_DIMS = [1, 2, 4]


@st.composite
def seq_inputs(draw, backend, max_len=5):
    """Generate (wIn, wRec, b, s0, elements, seq) for the general RNN cell.

    wIn:      [hidden, input] matrix
    wRec:     [hidden, hidden] matrix
    b:        [hidden] vector
    s0:       [hidden] vector (zeros)
    elements: list of [input] vectors
    """
    input_dim = draw(st.sampled_from(INPUT_DIMS))
    hidden_dim = draw(st.sampled_from(HIDDEN_DIMS))

    wIn = backend.random_matrix(draw, hidden_dim, input_dim)
    wRec = backend.random_matrix(draw, hidden_dim, hidden_dim)
    b = backend.random_vector(draw, hidden_dim)
    s0 = backend.zeros_vector(hidden_dim)

    n = draw(st.integers(1, max_len))
    elements = [backend.random_vector(draw, input_dim) for _ in range(n)]
    seq = _make_seq(elements)

    return wIn, wRec, b, s0, elements, seq, input_dim, hidden_dim


@st.composite
def tree_inputs(draw, backend, max_depth=3):
    """Generate (w, leaf_values, tree) for the tree RNN.

    w:      [hidden, input] matrix
    leaves: list of [input] vectors
    tree:   RTreeF structure with [input]-shaped leaves
    """
    input_dim = draw(st.sampled_from(INPUT_DIMS))
    hidden_dim = draw(st.sampled_from(HIDDEN_DIMS))

    w = backend.random_matrix(draw, hidden_dim, input_dim)
    leaves, tree = draw(_tree_recursive(backend, input_dim, max_depth))
    return w, leaves, tree, input_dim, hidden_dim


@st.composite
def _tree_recursive(draw, backend, dim, depth):
    if depth <= 0 or draw(st.booleans()):
        val = backend.random_vector(draw, dim)
        return [val], Left(val)
    else:
        left_vals, left_tree = draw(_tree_recursive(backend, dim, depth - 1))
        right_vals, right_tree = draw(_tree_recursive(backend, dim, depth - 1))
        return left_vals + right_vals, Right((left_tree, right_tree))


def _make_seq(elements):
    node = Left(())
    for x in reversed(elements):
        node = Right((x, node))
    return node
