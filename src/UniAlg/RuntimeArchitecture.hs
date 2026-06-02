{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Runtime (value-level) module builders for neural recursion schemes.

Parallel to the typeclass-dispatched builders in 'UniAlg.Architecture', but
take a value-level 'Shape'' descriptor instead of a type application @\@f@.
Use when the functor shape is known only at runtime (e.g. derived from a
grammar term at generation time).

The compile-time builders in 'UniAlg.Architecture' remain the preferred API
when @f@ is statically known.

Correspondence with compile-time API:

@
moduleR       ≈ (plain term definition, no recursion)
cataModuleR   ≈ cataModule  \@f
anaModuleR    ≈ anaModule   \@f
hyloModuleR   ≈ hyloModule  \@f
@
-}
module UniAlg.RuntimeArchitecture
  ( Shape'(..)
  , RAlg(..)
  , moduleR
  , cataModuleR
  , anaModuleR
  , hyloModuleR
  , rebuildAlg
  , rebuildBranch
  ) where

import Hydra.Phantoms (TTerm(..), unTTerm)
import qualified Hydra.Dsl.Terms as Terms
import Hydra.Dsl.Meta.Phantoms (var, (~>))
import Hydra.Kernel
  ( Definition(..)
  , FunctionType(..)
  , Module(..)
  , Name(..)
  , Namespace(..)
  , TermDefinition(..)
  , Type(..)
  , TypeScheme(..)
  )

import UniAlg.Core.Reduce (reduceTerm)
import UniAlg.Scheme.Internal (withSelf)
import UniAlg.Term.Internal
  ( tApply
  , tEither
  , tFst
  , tLam
  , tLeft
  , tPair
  , tRight
  , tSnd
  , tVar
  )


-- ── Value-level shape descriptor ─────────────────────────────────────────────

-- | Value-level descriptor of a polynomial endofunctor @F@.
--
-- Mirrors the 'UniAlg.Shape.Encode.Shape' typeclass and its instances as a
-- runtime value.  The correspondence is:
--
-- @
-- SUnit         ≈  Const ()
-- SConst        ≈  Const (TTerm k)      (any atom type)
-- SIdent        ≈  Identity             (recursive position)
-- SSum l r      ≈  Sum f g
-- SProd l r     ≈  Product f g
-- SExp s        ≈  Exp (TTerm i) f      (function position)
-- @
data Shape'
  = SUnit               -- ^ @Const ()@: no payload; used for base / unit cases.
  | SConst              -- ^ @Const A@: carries an external atom of any type.
  | SIdent              -- ^ @Identity@: a recursive child position.
  | SSum  Shape' Shape' -- ^ @Sum F G@: choice between two sub-functors.
  | SProd Shape' Shape' -- ^ @Product F G@: pair of sub-functors.
  | SExp  Shape'        -- ^ @Exp R F@: function-position; @R@ is the index type.
  deriving (Eq, Show)


-- ── Runtime algebra tree ─────────────────────────────────────────────────────

-- | Runtime algebra for a catamorphism over 'Shape''.
--
-- Mirrors the @Elim f a r@ type family in 'UniAlg.Architecture':
--
--   * @RAlgLeaf fn@ — a leaf branch.  @fn@ receives the flat list of slot
--     terms (left-to-right: 'SUnit' ↦ 0 slots, all others ↦ 1 slot each)
--     and returns the folded result.
--   * @RAlgNode l r@ — a sum node mirroring @'SSum'@; dispatches left/right.
data RAlg a
  = RAlgLeaf ([TTerm a] -> TTerm a)
  | RAlgNode (RAlg a) (RAlg a)


-- ── Slot extraction ───────────────────────────────────────────────────────────

-- | Extract slot terms from a branch shape, applying @step@ to each
-- recursive (@'SIdent'@) and function (@'SExp'@) position.
--
-- 'SUnit' contributes 0 slots.
-- 'SConst', 'SIdent', and 'SExp' each contribute 1 slot.
-- 'SProd' recurses left-then-right via @'tFst'@\/@'tSnd'@.
-- 'SSum' inside a product branch is a programming error.
collectSlotsR :: Shape' -> (TTerm a -> TTerm a) -> TTerm a -> [TTerm a]
collectSlotsR SUnit        _    _ = []
collectSlotsR SConst       _    x = [x]
collectSlotsR SIdent       step x = [step x]
collectSlotsR (SExp _)     step x = [tLam "inp" (step (tApply x (tVar "inp")))]
collectSlotsR (SProd l r)  step x =
  collectSlotsR l step (tFst x) ++ collectSlotsR r step (tSnd x)
collectSlotsR (SSum _ _)   _    _ =
  error "collectSlotsR: SSum found inside a product branch (malformed Shape')"


-- ── Algebra application ───────────────────────────────────────────────────────

-- | Apply a runtime algebra to a layer term, threading @step@ through
-- recursive positions.  Replaces @matchLayer \@f (alg . fmap step)@ from
-- 'UniAlg.Scheme.Internal.hyloT'.
applyAlgR :: Shape' -> (TTerm a -> TTerm a) -> RAlg a -> TTerm a -> TTerm a
applyAlgR (SSum l r)  step (RAlgNode lt rt) x =
  tEither
    (tLam "l" (applyAlgR l step lt (tVar "l")))
    (tLam "r" (applyAlgR r step rt (tVar "r")))
    x
applyAlgR shape       step (RAlgLeaf fn)    x = fn (collectSlotsR shape step x)
applyAlgR _           _    _               _ =
  error "applyAlgR: Shape'/RAlg structure mismatch (node count differs)"


-- ── Rebuild algebra (for anaModuleR) ─────────────────────────────────────────

-- | Construct an @'RAlg'@ that reconstructs the functor layer from its slot
-- terms.  Used as the fixed algebra in @'anaModuleR'@, mirroring
-- @buildLayer \@f@ in the compile-time API.
rebuildAlg :: Shape' -> RAlg a
rebuildAlg (SSum l r) = RAlgNode (rebuildAlg l) (rebuildAlg r)
rebuildAlg shape      = RAlgLeaf (rebuildBranch shape)

rebuildBranch :: Shape' -> [TTerm a] -> TTerm a
rebuildBranch shape slots = fst (go shape slots)
  where
    go SUnit       ss     = (TTerm Terms.unit, ss)
    go SConst      (x:ss) = (x, ss)
    go SIdent      (x:ss) = (x, ss)
    go (SExp _)    (x:ss) = (x, ss)
    go (SProd l r) ss     =
      let (lt, ss')  = go l ss
          (rt, ss'') = go r ss'
      in (tPair lt rt, ss'')
    go _ _ = error "rebuildBranch: shape/slots mismatch"


-- ── Runtime hyloT ────────────────────────────────────────────────────────────

-- | Runtime equivalent of 'UniAlg.Scheme.Internal.hyloT'.
--
-- Takes 'Shape'' instead of the typeclass-dispatched @\@f@.
-- Requires @?self@ to be bound by an enclosing 'withSelf' call.
hyloTR :: (?self :: TTerm a) => Shape' -> (TTerm a -> TTerm a) -> RAlg a -> TTerm a -> TTerm a
hyloTR shape coalg alg x = applyAlgR shape step alg (coalg x)
  where step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


-- ── Shared internals ──────────────────────────────────────────────────────────

-- | Type scheme for an n-argument function: @_a0 → … → _a{n-1}@.
-- Identical in purpose to the private @polyFnScheme@ in 'UniAlg.Architecture'.
rPolyFnScheme :: Int -> TypeScheme
rPolyFnScheme n = TypeScheme
  { typeSchemeVariables   = vars
  , typeSchemeBody        = foldr step ret (init tvars)
  , typeSchemeConstraints = Nothing
  }
  where
    vars  = [Name ("_a" <> show i) | i <- [0 .. n - 1]]
    tvars = fmap TypeVariable vars
    ret   = last tvars
    step a b = TypeFunction FunctionType
      { functionTypeDomain   = a
      , functionTypeCodomain = b
      }

hyloDefR
  :: String -> String -> [String]
  -> Shape'
  -> ([TTerm a] -> (TTerm a -> TTerm a, RAlg a))
  -> TermDefinition
hyloDefR ns name outerArgNames shape k = TermDefinition
  { termDefinitionName       = Name (ns <> "." <> name)
  , termDefinitionTerm       = reduceTerm $ foldr Terms.lambda (unTTerm innerTerm) outerArgNames
  , termDefinitionTypeScheme = Just (rPolyFnScheme (length outerArgNames + 2))
  }
  where
    vars         = map var outerArgNames
    (coalg, alg) = k vars
    appliedSelf  = foldl (\s n -> tApply s (var n)) (var (ns <> "." <> name)) outerArgNames
    innerTerm    = "x" ~> withSelf appliedSelf (hyloTR shape coalg alg (var "x"))


-- ── Public module builders ────────────────────────────────────────────────────

-- | Build a 'Module' for a non-recursive top-level function definition.
moduleR
  :: String      -- ^ Namespace (e.g. @"seed.generated"@)
  -> String      -- ^ Function name
  -> [Namespace] -- ^ Term dependencies
  -> [String]    -- ^ Outer parameter names (weights, biases, …)
  -> ([TTerm a] -> TTerm a)
  -> Module
moduleR ns name deps outerArgNames k = Module
  { moduleDescription      = Just "Non-recursive module definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm defn]
  }
  where
    defn = TermDefinition
      { termDefinitionName       = Name (ns <> "." <> name)
      , termDefinitionTerm       = reduceTerm $ foldr Terms.lambda (unTTerm body) outerArgNames
      , termDefinitionTypeScheme = Just (rPolyFnScheme (length outerArgNames + 1))
      }
    body = k (map var outerArgNames)

-- | Build a 'Module' for a catamorphism (fold) over a runtime 'Shape''.
--
-- Mirrors 'UniAlg.Architecture.cataModule'; passes @id@ as the coalgebra.
cataModuleR
  :: String -> String -> [Namespace] -> [String]
  -> Shape'
  -> ([TTerm a] -> RAlg a)
  -> Module
cataModuleR ns name deps outerArgNames shape k =
  hyloModuleR ns name deps outerArgNames shape $ \vs -> (id, k vs)

-- | Build a 'Module' for an anamorphism (unfold) over a runtime 'Shape''.
--
-- The coalgebra @k@ should build a functor-layer TTerm from the seed,
-- typically via @'rebuildAlg'@-compatible encoding.
-- Mirrors 'UniAlg.Architecture.anaModule'; uses @'rebuildAlg' shape@ as the
-- fixed fold algebra.
anaModuleR
  :: String -> String -> [Namespace] -> [String]
  -> Shape'
  -> ([TTerm a] -> (TTerm a -> TTerm a))
  -> Module
anaModuleR ns name deps outerArgNames shape k =
  hyloModuleR ns name deps outerArgNames shape $ \vs ->
    (k vs, rebuildAlg shape)

-- | Build a 'Module' for a hylomorphism over a runtime 'Shape''.
--
-- Mirrors 'UniAlg.Architecture.hyloModule'.
hyloModuleR
  :: String -> String -> [Namespace] -> [String]
  -> Shape'
  -> ([TTerm a] -> (TTerm a -> TTerm a, RAlg a))
  -> Module
hyloModuleR ns name deps outerArgNames shape k = Module
  { moduleDescription      = Just "Runtime hylomorphic recursive definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm (hyloDefR ns name outerArgNames shape k)]
  }
