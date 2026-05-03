# Simple Transformer — unified-algebra DSL
#
# Categorical reading (CDL paper, Figure 1):
#   A transformer block is a parametric endofunctor algebra homomorphism.
#   Stacking N blocks is a fold over List(embed), functor 1 + embed & X.
#   Weight tying across blocks = precomposing with the copy map Δ_P.

import numpy

# --- Custom activation ---
define unary gelu(x) = x * sigmoid(1.702 * x)

# --- Algebra ---
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)

# --- Sort ---
spec embed(real, axes=[feature:8])

# --- Ops (each declared once — called with different weights at runtime) ---

# Linear projection: the parametric morphism (P, embed) -> embed
op proj : embed -> embed
  einsum = "ij,j->i"
  algebra = real

# Elementwise combine: embed x embed -> embed (scoring, gating)
op hadamard : embed -> embed
  einsum = "i,i->i"
  algebra = real

# Pointwise activations
op softmax_act : embed -> embed
  nonlinearity = softmax

op gelu_act : embed -> embed
  nonlinearity = gelu

# --- Cells ---

# Attention softmax gate (trivial cell, but named for the pipeline)
cell attn_gate : embed -> embed = softmax_act

# FFN activation (single GELU between two projections)
cell ffn_act : embed -> embed = gelu_act

# --- Structural: list functor for N-layer stacking ---
# F(X) = 1 + embed × X  (the List endofunctor, Example 2.9 in CDL)
functor Layers : 1 + embed & X
