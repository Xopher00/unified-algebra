"""
Compare unialg-generated RNN implementations against TensorFlow references.

fold_rnn: s_t = multiply(w, x_t) + s_{t-1}   (linear recurrence, right-fold)
tree_rnn: leaf(a) -> w*a, node(l,r) -> l + r  (weighted leaf sum)

TF references use tf.foldl and tf.reduce_sum to compute the same quantities
independently of the unialg codegen, providing a numerical ground truth.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Suppress TF startup noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Generated modules written by NeuralRecursionCodegenTest
sys.path.insert(0, "/tmp/unialg-neural-fold")
sys.path.insert(0, "/tmp/unialg-neural-tree")

from hydra.dsl.python import Left, Right
from neural.fold_rnn import fold_rnn
from neural.tree_rnn import tree_rnn


# ── Constructors for Either-encoded structures ────────────────────────────────

def make_seq(xs):
    """Encode a Python list as a right-associated cons list.
    [] -> Left(())
    [x, ...] -> Right((x, make_seq(...)))
    """
    result = Left(())
    for x in reversed(xs):
        result = Right((x, result))
    return result


def make_leaf(a):
    return Left(a)


def make_node(l, r):
    return Right((l, r))


# ── fold_rnn vs tf.foldl ──────────────────────────────────────────────────────
#
# fold_rnn(w, s0, [a1,a2,a3]) = w*a1 + (w*a2 + (w*a3 + s0))
#                              = w*(a1+a2+a3) + s0   (for scalars)
#
# tf.foldl with fn = lambda acc, x: w*x + acc gives the same result
# (left-fold and right-fold coincide here because + and * are associative/
# commutative for real-valued scalars and element-wise tensor ops).

w    = np.float32(2.0)
s0   = np.float32(0.0)
xs   = [np.float32(v) for v in [1.0, 2.0, 3.0]]

gen_fold = float(fold_rnn(w, s0, make_seq(xs)))

tf_fold = float(tf.foldl(
    fn=lambda acc, x: tf.add(tf.multiply(tf.constant(w), x), acc),
    elems=tf.constant(xs),
    initializer=tf.constant(s0),
))

assert abs(gen_fold - tf_fold) < 1e-5, (
    f"fold_rnn mismatch: generated={gen_fold:.6f}, tf={tf_fold:.6f}"
)
print(f"PASS fold_rnn vs tf.foldl:      generated={gen_fold:.4f}, tf={tf_fold:.4f}")


# ── tree_rnn vs tf.reduce_sum(w * leaves) ────────────────────────────────────
#
# tree_rnn(w, leaf(a)) = w*a
# tree_rnn(w, node(l,r)) = tree_rnn(w,l) + tree_rnn(w,r)
#
# By linearity: tree_rnn(w, tree) = w * sum_of_all_leaf_values
# TF reference: tf.multiply(w, tf.reduce_sum(leaf_vals))

w_t       = np.float32(3.0)
leaf_vals = [np.float32(v) for v in [1.0, 2.0, 3.0, 4.0]]

tree_in = make_node(
    make_node(make_leaf(leaf_vals[0]), make_leaf(leaf_vals[1])),
    make_node(make_leaf(leaf_vals[2]), make_leaf(leaf_vals[3])),
)

gen_tree = float(tree_rnn(w_t, tree_in))

tf_tree = float(tf.multiply(
    tf.constant(w_t),
    tf.reduce_sum(tf.constant(leaf_vals)),
))

assert abs(gen_tree - tf_tree) < 1e-5, (
    f"tree_rnn mismatch: generated={gen_tree:.6f}, tf={tf_tree:.6f}"
)
print(f"PASS tree_rnn vs tf.reduce_sum: generated={gen_tree:.4f}, tf={tf_tree:.4f}")


# ── Vector inputs: fold_rnn over sequences of 1-D arrays ─────────────────────
#
# Same formula, but w, s0, and each x_i are 1-D numpy arrays.
# TF reference uses tf.foldl over a 2-D tensor (axis-0 = time steps).

w_v  = np.array([1.0, 2.0], dtype=np.float32)
s0_v = np.array([0.0, 0.0], dtype=np.float32)
xs_v = [
    np.array([1.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
    np.array([1.0, 1.0], dtype=np.float32),
]

gen_fold_v = fold_rnn(w_v, s0_v, make_seq(xs_v))

tf_fold_v = tf.foldl(
    fn=lambda acc, x: tf.add(tf.multiply(tf.constant(w_v), x), acc),
    elems=tf.stack(xs_v),
    initializer=tf.constant(s0_v),
).numpy()

assert np.allclose(gen_fold_v, tf_fold_v, atol=1e-5), (
    f"fold_rnn (vector) mismatch: generated={gen_fold_v}, tf={tf_fold_v}"
)
print(f"PASS fold_rnn (vector) vs tf.foldl: generated={gen_fold_v}, tf={tf_fold_v}")


# ── Matrix inputs: fold_rnn over sequences of 2-D arrays ─────────────────────
#
# w, s0, and each x_i are 2×2 matrices.
# fold_rnn(w, s0, [x1,x2,x3]) = w⊙x1 + (w⊙x2 + (w⊙x3 + s0))   (element-wise)
#                              = w⊙(x1+x2+x3) + s0
# TF reference: tf.foldl with fn = lambda acc, x: w⊙x + acc

w_m  = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
s0_m = np.zeros((2, 2), dtype=np.float32)
xs_m = [
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
]

gen_fold_m = fold_rnn(w_m, s0_m, make_seq(xs_m))

tf_fold_m = tf.foldl(
    fn=lambda acc, x: tf.add(tf.multiply(tf.constant(w_m), x), acc),
    elems=tf.stack(xs_m),
    initializer=tf.constant(s0_m),
).numpy()

assert np.allclose(gen_fold_m, tf_fold_m, atol=1e-5), (
    f"fold_rnn (matrix) mismatch:\ngenerated=\n{gen_fold_m}\ntf=\n{tf_fold_m}"
)
print(f"PASS fold_rnn (matrix) vs tf.foldl:\n{gen_fold_m}")


# ── Matrix inputs: tree_rnn over a balanced tree of 2-D arrays ───────────────
#
# w is a 2×2 matrix; leaves are 2×2 matrices.
# tree_rnn(w, leaf(a)) = w⊙a
# tree_rnn(w, node(l,r)) = tree_rnn(w,l) + tree_rnn(w,r)
#
# By linearity: result = w⊙(sum of all leaf matrices)
# TF reference: tf.multiply(w, tf.reduce_sum(leaf_stack, axis=0))

w_tm = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
leaf_mats = [
    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
]

tree_mat_in = make_node(
    make_node(make_leaf(leaf_mats[0]), make_leaf(leaf_mats[1])),
    make_node(make_leaf(leaf_mats[2]), make_leaf(leaf_mats[3])),
)

gen_tree_m = tree_rnn(w_tm, tree_mat_in)

tf_tree_m = tf.multiply(
    tf.constant(w_tm),
    tf.reduce_sum(tf.stack(leaf_mats), axis=0),
).numpy()

assert np.allclose(gen_tree_m, tf_tree_m, atol=1e-5), (
    f"tree_rnn (matrix) mismatch:\ngenerated=\n{gen_tree_m}\ntf=\n{tf_tree_m}"
)
print(f"PASS tree_rnn (matrix) vs tf.reduce_sum:\n{gen_tree_m}")


print("\nAll TF comparisons passed.")
