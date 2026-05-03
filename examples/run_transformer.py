"""Simple Transformer — parse, compile, execute.

Architecture (single-head, single vector, d=8):
  1. Self-attention: Q,K,V projections -> score -> softmax -> mix
  2. Residual connection: attn(x) + x
  3. FFN: up-project -> gelu -> down-project
  4. Residual connection: ffn(h) + h

All architecture declared in transformer.ua.
Python only parses, creates weights, and calls entry points.

Categorical reading: each call to prog("proj", W, x) is a parametric
morphism (P, A) -> B in the 2-category Para. The residual connection
x + f(x) is the semiring plus (real.plus = add).
"""
import numpy as np
from unialg import parse_ua

from pathlib import Path
with open(Path(__file__).parent / "transformer.ua") as f:
    prog = parse_ua(f.read())

print("Entry points:", prog.entry_points())

d = 8
np.random.seed(42)
W_q = np.random.randn(d, d) * 0.1
W_k = np.random.randn(d, d) * 0.1
W_v = np.random.randn(d, d) * 0.1
W_up = np.random.randn(d, d) * 0.1
W_down = np.random.randn(d, d) * 0.1

x = np.random.randn(d)
print(f"Input: {x}")

# --- Self-Attention ---
# Three calls to the same 'proj' morphism with different parameter tensors
q = prog("proj", W_q, x)
k = prog("proj", W_k, x)
v = prog("proj", W_v, x)

scores = prog("hadamard", q, k)
attn_weights = prog("attn_gate", scores)
context = prog("hadamard", attn_weights, v)

# Residual: semiring plus
h = context + x
print(f"After attention + residual: {h}")

# --- FFN ---
h_up = prog("proj", W_up, h)
h_act = prog("ffn_act", h_up)
h_down = prog("proj", W_down, h_act)

# Residual
output = h_down + h
print(f"Output: {output}")

# --- Oracle verification ---
def gelu_np(x):
    return x * (1 / (1 + np.exp(-1.702 * x)))

q_o = W_q @ x
k_o = W_k @ x
v_o = W_v @ x
sc = q_o * k_o
sw = np.exp(sc - sc.max()); sw /= sw.sum()
ctx = sw * v_o
h_o = ctx + x
act_o = gelu_np(W_up @ h_o)
out_o = (W_down @ act_o) + h_o

np.testing.assert_allclose(q, q_o, rtol=1e-6)
np.testing.assert_allclose(k, k_o, rtol=1e-6)
np.testing.assert_allclose(v, v_o, rtol=1e-6)
np.testing.assert_allclose(attn_weights, sw, rtol=1e-6)
np.testing.assert_allclose(context, ctx, rtol=1e-6)
np.testing.assert_allclose(h_act, gelu_np(W_up @ h_o), rtol=1e-6)
np.testing.assert_allclose(output, out_o, rtol=1e-6)
print("\nAll 7 assertions passed.")
