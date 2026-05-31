{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Turn a functor + canonical (co)algebra into a codegen-ready 'Module'.

=== Type-application bridge

'hyloModule' needs @f@ as a compile-time type with a term-level 'Shape' instance.
The 'PolyF' AST is a value. Resolution:

- __Seed MVP__ (this module): hand-pair each seed 'PolyF' with its concrete
  @hyloModule \@T@ call in 'Seed' — finite, explicit list.
- __General depth-N__ (future): a Template Haskell splice consuming
  'enumerate' and emitting one @hyloModule \@\<typeFromPolyF\>@ per AST.

=== Canonical architectures

Two canonical forms per functor:

1. __Structural identity__ @(id, buildLayer)@: coalg=id, alg reassembles
   each layer. The generated function is an identity transform.
2. __Op-family algebra__: generic algebra derived from the 'PolyF' AST.
   @Const ()@ → base seed; @Identity@ → recurse; @Product@ → combine
   children with a binary op; @Sum@ → branch.

The seed set in 'Seed' uses these patterns.
-}
module Generate
  ( classifyPolyF
  , matchesSeed
  ) where

import Grammar
import Seed
import Catalogue (seeds)


-- | Which seed entry (if any) a 'PolyF' AST corresponds to.
matchesSeed :: PolyF -> Maybe SeedEntry
matchesSeed poly = case filter (\s -> seedPolyF s == poly) seeds of
  (s:_) -> Just s
  []    -> Nothing
  where
    seedPolyF s = case seedLabel s of
      "seqCata"    -> KUnit :+: (KConst :*: Hole)
      "treeCata"   -> KConst :+: (Hole :*: Hole)
      "streamAna"  -> KConst :*: Hole
      "mooreAna"   -> KConst :*: ExpF Hole
      _            -> KUnit


-- | Classify a 'PolyF' by its architecture role.
classifyPolyF :: PolyF -> ArchClass
classifyPolyF poly
  | polyArity poly == 0 = NoStructure
  | hasExpF poly         = AnaArch
  | hasSumBase poly      = CataArch
  | otherwise            = HyloArch
  where
    hasExpF (ExpF _)  = True
    hasExpF (a :+: b) = hasExpF a || hasExpF b
    hasExpF (a :*: b) = hasExpF a || hasExpF b
    hasExpF _         = False

    hasSumBase (KUnit :+: _) = True
    hasSumBase (_ :+: _)     = True
    hasSumBase _             = False
