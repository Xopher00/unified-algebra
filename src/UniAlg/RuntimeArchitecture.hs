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
  , RCoElim(..)
  , rcoElimToTerm
  , moduleR
  , cataDefR
  , cataModuleR
  , anaDefR
  , anaModuleR
  , hyloDefR
  , hyloModuleR
  , rebuildAlg
  , rebuildBranch
  ) where

import Hydra.Phantoms (TTerm(..), unTTerm)
import qualified Hydra.Dsl.Terms as Terms
import Hydra.Dsl.Meta.Phantoms (var, (~>))
import Hydra.Kernel
  ( Definition(..)
  , Module(..)
  , Namespace(..)
  , TermDefinition
  )

import UniAlg.Architecture (hyloDefWith, plainDefWith)
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


-- ── Runtime co-eliminator (value-level CoElim) ──────────────────────────────

-- | Runtime equivalent of 'UniAlg.Shape.Encode.CoElim'.
--
-- Mirrors the structure of 'RAlg': a leaf carries the flat slot list a
-- non-sum branch must produce, and a node selects a sum branch.
--
--   * @RCoElimLeaf slots@ — the user's coalgebra has produced the slot
--     list expected by 'rebuildBranch' for the surrounding non-sum shape.
--   * @RCoElimNode (Left v)@ / @RCoElimNode (Right v)@ — sum dispatch,
--     mirroring 'SSum' / 'RAlgNode'.
--
-- Used by 'hyloDefR' / 'anaDefR' so that runtime coalgebras can return an
-- explicit layer value (parallel to the static side's 'CoElim'), rather
-- than an opaque @'TTerm' a@ that 'applyAlgR' would have to decode.
data RCoElim a
  = RCoElimLeaf [TTerm a]
  | RCoElimNode (Either (RCoElim a) (RCoElim a))


-- | Encode an 'RCoElim' value back into the @'TTerm'@ representation that
-- 'applyAlgR' consumes.  Sum branches wrap via 'tLeft' / 'tRight'; leaf
-- branches delegate to 'rebuildBranch' at the surrounding shape.
rcoElimToTerm :: Shape' -> RCoElim a -> TTerm a
rcoElimToTerm shape       (RCoElimLeaf slots)        = rebuildBranch shape slots
rcoElimToTerm (SSum l _)  (RCoElimNode (Left v))     = tLeft  (rcoElimToTerm l v)
rcoElimToTerm (SSum _ r)  (RCoElimNode (Right v))    = tRight (rcoElimToTerm r v)
rcoElimToTerm _           _                          =
  error "rcoElimToTerm: Shape' / RCoElim structural mismatch"


-- ── Runtime cata / ana / hylo ────────────────────────────────────────────────

-- | Runtime catamorphism over 'Shape''.  Equivalent to
-- 'UniAlg.Scheme.Internal.cataT', value-level dispatch.  No coalgebra
-- phase: the input @x@ is already a 'TTerm' encoding the structure.
cataTR :: (?self :: TTerm a) => Shape' -> RAlg a -> TTerm a -> TTerm a
cataTR shape alg x = applyAlgR shape step alg x
  where step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


-- | Runtime anamorphism over 'Shape''.  Equivalent to
-- 'UniAlg.Scheme.Internal.anaT', value-level dispatch.  Fixed algebra is
-- @'rebuildAlg' shape@.
anaTR :: (?self :: TTerm a) => Shape' -> (TTerm a -> RCoElim a) -> TTerm a -> TTerm a
anaTR shape coalg = hyloTR shape coalg (rebuildAlg shape)


-- | Runtime hylomorphism over 'Shape''.  Equivalent to
-- 'UniAlg.Scheme.Internal.hyloT', value-level dispatch.
--
-- The coalgebra returns an explicit 'RCoElim' value; 'rcoElimToTerm'
-- re-encodes it into the @'TTerm'@ representation that 'applyAlgR'
-- consumes.  Recursive positions are replaced with self-calls before the
-- algebra is applied.
--
-- Requires @?self@ to be bound by an enclosing 'withSelf' call.
hyloTR :: (?self :: TTerm a) => Shape' -> (TTerm a -> RCoElim a) -> RAlg a -> TTerm a -> TTerm a
hyloTR shape coalg alg x = applyAlgR shape step alg (rcoElimToTerm shape (coalg x))
  where step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


-- ── Single-definition builders (cataDefR / anaDefR / hyloDefR) ──────────────

-- | Build a recursive 'TermDefinition' for a runtime catamorphism.
--
-- Lower-level than 'cataModuleR'; use when combining multiple
-- definitions in one module.  Mirrors 'UniAlg.Architecture.cataDef'.
cataDefR
  :: forall a
   . String -> String -> [String]
  -> Shape'
  -> ([TTerm a] -> RAlg a)
  -> TermDefinition
cataDefR ns name outerArgNames shape k =
  hyloDefWith ns name outerArgNames $ \self vs ->
    withSelf self (cataTR shape (k vs) (var "x"))


-- | Build a recursive 'TermDefinition' for a runtime anamorphism.
--
-- Lower-level than 'anaModuleR'.  Mirrors 'UniAlg.Architecture.anaDef'.
-- The user's coalgebra returns an explicit 'RCoElim' value; the fixed
-- algebra is @'rebuildAlg' shape@.
anaDefR
  :: forall a
   . String -> String -> [String]
  -> Shape'
  -> ([TTerm a] -> (TTerm a -> RCoElim a))
  -> TermDefinition
anaDefR ns name outerArgNames shape k =
  hyloDefWith ns name outerArgNames $ \self vs ->
    withSelf self (anaTR shape (k vs) (var "x"))


-- | Build a recursive 'TermDefinition' for a runtime hylomorphism.
--
-- Lower-level than 'hyloModuleR'.  Mirrors 'UniAlg.Architecture.hyloDef'.
-- The coalgebra returns an explicit 'RCoElim' value (re-encoded via
-- 'rcoElimToTerm' before 'applyAlgR' consumes it).
hyloDefR
  :: forall a
   . String -> String -> [String]
  -> Shape'
  -> ([TTerm a] -> (TTerm a -> RCoElim a, RAlg a))
  -> TermDefinition
hyloDefR ns name outerArgNames shape k =
  hyloDefWith ns name outerArgNames $ \self vs ->
    let (coalg, alg) = k vs
    in withSelf self (hyloTR shape coalg alg (var "x"))


-- ── Public module builders ────────────────────────────────────────────────────

-- | Build a 'Module' for a non-recursive top-level function definition.
moduleR
  :: forall a
   . String      -- ^ Namespace (e.g. @"seed.generated"@)
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
  , moduleDefinitions      = [DefinitionTerm (plainDefWith ns name outerArgNames k)]
  }

-- | Build a 'Module' for a catamorphism (fold) over a runtime 'Shape''.
--
-- Mirrors 'UniAlg.Architecture.cataModule'.
cataModuleR
  :: String -> String -> [Namespace] -> [String]
  -> Shape'
  -> ([TTerm a] -> RAlg a)
  -> Module
cataModuleR ns name deps outerArgNames shape k = Module
  { moduleDescription      = Just "Runtime catamorphic recursive definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm (cataDefR ns name outerArgNames shape k)]
  }

-- | Build a 'Module' for an anamorphism (unfold) over a runtime 'Shape''.
--
-- The coalgebra returns an explicit 'RCoElim' value describing the
-- functor layer to build.  Mirrors 'UniAlg.Architecture.anaModule'; uses
-- @'rebuildAlg' shape@ as the fixed fold algebra.
anaModuleR
  :: String -> String -> [Namespace] -> [String]
  -> Shape'
  -> ([TTerm a] -> (TTerm a -> RCoElim a))
  -> Module
anaModuleR ns name deps outerArgNames shape k = Module
  { moduleDescription      = Just "Runtime anamorphic recursive definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm (anaDefR ns name outerArgNames shape k)]
  }

-- | Build a 'Module' for a hylomorphism over a runtime 'Shape''.
--
-- Mirrors 'UniAlg.Architecture.hyloModule'.
hyloModuleR
  :: String -> String -> [Namespace] -> [String]
  -> Shape'
  -> ([TTerm a] -> (TTerm a -> RCoElim a, RAlg a))
  -> Module
hyloModuleR ns name deps outerArgNames shape k = Module
  { moduleDescription      = Just "Runtime hylomorphic recursive definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm (hyloDefR ns name outerArgNames shape k)]
  }
