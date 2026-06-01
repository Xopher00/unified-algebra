{-|
Value-level polynomial functor AST and bounded enumeration.

The lowering-eligible grammar is @{Const, Identity, Sum, Product, Exp}@.
Raw @[]@ is excluded (non-lowering). 'PolyF' represents this grammar as a
value-level ADT so the explore layer can enumerate and classify functors
programmatically.

Each 'PolyF' atom has a direct Haskell type counterpart in "UniAlg.Shape":

@
PolyF atom   Haskell type           Pretty
-----------  ---------------------  ------
KUnit        Const ()               1
KConst       Const (TTerm a)        A
Hole         Identity               X
a :+: b      Sum f g                (a + b)
a :*: b      Product f g            (a × b)
ExpF a       Exp (TTerm i)          (I → a)
@

'polyArity' counts the number of 'Hole' positions; it equals the number of
recursive children each node has (0 for leaves, 1 for lists, 2 for binary trees, …).

'enumerate' generates all well-formed 'PolyF' up to a given depth.
-}
module Grammar
  ( PolyF(..)
  , enumerate
  , polyArity
  , prettyPolyF
  ) where


-- | Value-level representation of the polynomial functor grammar.
data PolyF
  = KUnit              -- ^ @Const ()@ — unit / base case
  | KConst             -- ^ @Const (TTerm a)@ — carries a value
  | Hole               -- ^ @Identity@ — the recursive position
  | PolyF :+: PolyF    -- ^ @Sum@ — coproduct / branching
  | PolyF :*: PolyF    -- ^ @Product@ — product / pairing
  | ExpF PolyF         -- ^ @Exp (TTerm i)@ — exponential / reader
  deriving (Eq, Ord, Show)

infixl 6 :+:
infixl 7 :*:


-- | Enumerate all well-formed 'PolyF' up to a given depth.
--
-- Depth 0: atoms (@KUnit@, @KConst@, @Hole@).
-- Depth N: all binary combinations of depth @N-1@ trees + @ExpF@ of depth @N-1@.
enumerate :: Int -> [PolyF]
enumerate 0 = [KUnit, KConst, Hole]
enumerate n =
  let prev = enumerate (n - 1)
      sums = [a :+: b | a <- prev, b <- prev]
      prods = [a :*: b | a <- prev, b <- prev]
      exps = [ExpF a | a <- prev]
  in prev ++ sums ++ prods ++ exps


-- | Count the number of recursive positions ('Hole') in a functor.
polyArity :: PolyF -> Int
polyArity KUnit     = 0
polyArity KConst    = 0
polyArity Hole      = 1
polyArity (a :+: b) = polyArity a + polyArity b
polyArity (a :*: b) = polyArity a + polyArity b
polyArity (ExpF a)  = polyArity a


-- | Human-readable rendering of a functor shape.
prettyPolyF :: PolyF -> String
prettyPolyF KUnit     = "1"
prettyPolyF KConst    = "A"
prettyPolyF Hole      = "X"
prettyPolyF (a :+: b) = "(" ++ prettyPolyF a ++ " + " ++ prettyPolyF b ++ ")"
prettyPolyF (a :*: b) = "(" ++ prettyPolyF a ++ " × " ++ prettyPolyF b ++ ")"
prettyPolyF (ExpF a)  = "(I → " ++ prettyPolyF a ++ ")"
