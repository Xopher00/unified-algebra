{-# LANGUAGE TemplateHaskell #-}

{-|
Auto-generated backend op surface.

Every op present in the canonical backend spec (@backends\/numpy.json@) is
surfaced here as a typed Haskell binding:

* Arity 1 (e.g. @tanh@):   @TTerm a -> TTerm a@
* Arity 2 (e.g. @add@):    @TTerm a -> TTerm a -> TTerm a@
* Arity 3 (e.g. @clip@):   @TTerm a -> TTerm a -> TTerm a -> TTerm a@

Dotted keys (@reduce.*@, @structural.*@) are mangled to camelCase:
@reduceAdd@, @structuralExpandDims@, etc.

=== Registry resolver

'op' resolves any raw spec key (including dotted forms) to its symbolic
base 'TTerm'.  Used by the tensor contraction compiler for dynamic,
semiring-driven op selection.  Errors on unknown keys.

@
-- Static user-facing surface (generated):
result = add a b
result = tanh x

-- Dynamic internal use (contraction compiler):
import UniAlg.Core.Ops (op)
product_ = op timesKey @@ a @@ b
@
-}
module UniAlg.Core.Ops where

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

import Hydra.Kernel (Name(..), Term(..))
import Hydra.Phantoms (TTerm(..))

import UniAlg.Core.Ops.Generate
  ( genBackendOps
  , op1
  , op2
  , op3
  , opBase
  )

$(genBackendOps "backends/numpy.json")


-- | Resolve a raw backend op key to its symbolic base 'TTerm'.
--
-- Covers all keys in @backends\/numpy.json@, including dotted forms:
-- @op \"add\"@, @op \"reduce.add\"@, @op \"structural.expand_dims\"@.
--
-- Errors at construction time if the key is absent from the canonical spec
-- (fail-fast, rather than deferring to a lowering miss).
op :: String -> TTerm a
op key =
  TTerm $ Map.findWithDefault
    (error ("op: unknown backend op key " <> show key
            <> " — check backends/numpy.json"))
    key
    opRegistry
