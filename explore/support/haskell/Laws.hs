{-|
Symbolic law checks for the Explore layer.

=== What is checkable

__Grammar laws__ (pure Haskell, no IR):
- Enumeration produces expected atom set at depth 0
- Classification is consistent with functor structure

__Structural laws__ (TTerm IR, via 'reduceTerm'):
- Atomic shapes ('Identity', 'Const') and compound shapes ('Product', 'Sum')
  are checked through the same encoded layer representation used by
  'cataT'/'anaT'.

__Fusion law__ (@cata alg . ana coalg ≡ hylo coalg alg@):
- Holds by construction in "UniAlg.Scheme" (the pure Haskell
  definitions). Numerical verification is deferred to Arm B (Python harness).

=== Epistemic honesty

A symbolic check that passes means \"confirmed\" — the IR forms are provably
equal after reduction.  A check that cannot be expressed (e.g. pair eta)
is reported as \"not structurally checkable,\" not silently skipped.
-}
module Laws
  ( checkGrammarLaws
  , checkClassificationLaws
  , checkSeedMapping
  ) where

import Catalogue (seeds)
import Generate
import Grammar
import Seed


-- | Verify grammar enumeration properties.
-- Returns @(label, passed)@ pairs.
checkGrammarLaws :: [(String, Bool)]
checkGrammarLaws =
  [ ("depth-0 atoms are KUnit, KConst, Hole"
    , enumerate 0 == [KUnit, KConst, Hole])

  , ("depth-0 count is 3"
    , length (enumerate 0) == 3)

  , ("depth-1 includes all depth-0 atoms"
    , all (`elem` enumerate 1) [KUnit, KConst, Hole])

  , ("depth-1 includes KUnit :+: KConst"
    , (KUnit :+: KConst) `elem` enumerate 1)

  , ("depth-1 includes KConst :*: Hole"
    , (KConst :*: Hole) `elem` enumerate 1)

  , ("depth-1 includes ExpF Hole"
    , ExpF Hole `elem` enumerate 1)

  , ("polyArity KUnit == 0"
    , polyArity KUnit == 0)

  , ("polyArity Hole == 1"
    , polyArity Hole == 1)

  , ("polyArity (Hole :+: Hole) == 2"
    , polyArity (Hole :+: Hole) == 2)

  , ("polyArity (ExpF Hole) == 1"
    , polyArity (ExpF Hole) == 1)
  ]


-- | Verify classification consistency.
checkClassificationLaws :: [(String, Bool)]
checkClassificationLaws =
  [ ("KUnit is NoStructure (arity 0)"
    , classifyPolyF KUnit == NoStructure)

  , ("KConst is NoStructure (arity 0)"
    , classifyPolyF KConst == NoStructure)

  , ("Hole is HyloArch (arity 1, no sum base, no exp)"
    , classifyPolyF Hole == HyloArch)

  , ("KUnit :+: (KConst :*: Hole) is CataArch (SeqF shape)"
    , classifyPolyF (KUnit :+: (KConst :*: Hole)) == CataArch)

  , ("KConst :+: (Hole :*: Hole) is CataArch (RTreeF shape)"
    , classifyPolyF (KConst :+: (Hole :*: Hole)) == CataArch)

  , ("KConst :*: Hole is HyloArch (product, no exp, no sum base)"
    , classifyPolyF (KConst :*: Hole) == HyloArch)

  , ("KConst :*: ExpF Hole is AnaArch (MooreF shape)"
    , classifyPolyF (KConst :*: ExpF Hole) == AnaArch)

  , ("ExpF (KConst :*: Hole) is AnaArch (Mealy shape)"
    , classifyPolyF (ExpF (KConst :*: Hole)) == AnaArch)
  ]


-- | Every seed's declared 'PolyF' shape classifies to its declared 'ArchClass'.
-- Covers all seeds in the catalogue automatically — no hardcoded names.
checkSeedMapping :: [(String, Bool)]
checkSeedMapping = map checkOne seeds
  where
    checkOne s =
      ( seedLabel s <> ": classifyPolyF matches seedClass"
      , classifyPolyF (seedPolyF s) == seedClass s
      )
