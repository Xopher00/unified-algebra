{-# LANGUAGE OverloadedStrings #-}

{-|
Architecture functor aliases for the Explore layer.

This module defines the polynomial-functor shapes that correspond to named
neural-network architectures.  Each alias is a saturated type synonym built
from the fundamental building blocks in "UniAlg.Semantics.Functors".

=== Recursion-scheme grammar

All aliases here lower through Hydra (use @'recModule' \@F@):

+----------+---+------------------------------------------+-------------------+
| Alias    |   | F(X)                                     | Architecture      |
+----------+---+------------------------------------------+-------------------+
| SeqF a   | = | 1 + (a × X)                              | Folding RNN       |
| RTreeF a | = | a + (X × X)                              | Recursive NN      |
| StreamF o| = | o × X                                    | Unfolding RNN     |
| MooreF o i=  | o × (i → X)                              | Moore machine     |
+----------+---+------------------------------------------+-------------------+

The @i →@ slot in 'MooreF' is handled by the 'Exp' functor from
"UniAlg.Semantics.Functors".

@SeqF@, @RTreeF@, and @StreamF@ are the polynomial encodings of list, binary
tree, and stream that avoid the non-lowering @[]@\/'RoseF' paths.
-}
module Explore.Archs
  ( SeqF
  , RTreeF
  , StreamF
  , MooreF
  ) where

import UniAlg


-- | @F(X) = 1 + (a × X)@  — sequence, RNN layer, or transformer block.
-- @InL (Const ())@ is the empty sequence; @InR (Pair (Const a) (Identity rest))@
-- carries one @'TTerm' a@ layer and the tail.
type SeqF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

-- | @F(X) = a + (X × X)@  — binary tree with data at leaves.
-- @InL (Const a)@ is a leaf carrying @'TTerm' a@; @InR (Pair l r)@ is an
-- internal node with two recursive subtrees.
type RTreeF a = Sum (Const (TTerm a)) (Product Identity Identity)

-- | @F(X) = o × X@  — stream or unfolding RNN.
-- Each step emits @'TTerm' o@ and continues with the next state.
type StreamF o = Product (Const (TTerm o)) Identity

-- | @F(X) = o × (i → X)@  — Moore machine.
-- The coalgebra maps each state to a @(output, i→next-state)@ pair.
type MooreF o i = Product (Const (TTerm o)) (Exp (TTerm i))
