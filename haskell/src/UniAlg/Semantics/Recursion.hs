{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ImplicitParams      #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

{-|
Recursion schemes at two levels, plus the 'Module' constructors that arch
authors call directly.

=== Haskell level

'Fix', 'cata', 'ana', and 'hylo' are the standard recursion schemes over
real Haskell values.

=== TTerm level (code generation)

'cataT', 'anaT', and 'hyloT' operate on symbolic 'TTerm' values and emit
Hydra IR.  The key difference from their Haskell counterparts is
self-reference.  A recursive Python function must call itself by name, but
at code-generation time the name is not yet bound.  'withSelf' injects it as
an implicit parameter @?self :: 'TTerm' a@ so that each recursive step emits
the correct self-application.

=== Module constructors

'recModule' handles 'withSelf' and the partial self-application automatically.
'cataModule' and 'anaModule' are the arch-author-facing wrappers — they accept
natural handler types ('Elim', 'CoElim') so that callers never see
@Data.Functor.*@ constructors.  Spec authors never call 'withSelf' directly.

@
cataModule \@(SeqF Tensor) \"seed.seq\" \"fold_seq\"
           [Namespace \"numpy\"] [\"w\", \"s0\"] $ \\[w, s0] ->
  ( s0
  , \\a s -> add (contraction real \"hi,i->h\" w a) s
  )
@
-}
module UniAlg.Semantics.Recursion
  ( -- * Haskell-level schemes
    Fix(..)
  , cata
  , ana
  , hylo

    -- * TTerm-level schemes
  , withSelf
  , cataT
  , anaT
  , hyloT

    -- * Module constructors
  , recDef
  , recModule
  , cataModule
  , anaModule
  ) where

import Hydra.Phantoms (TTerm(..), unTTerm)
import qualified Hydra.Dsl.Terms as Terms

import Hydra.Dsl.Meta.Phantoms
  ( var
  , (~>)
  )

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

import UniAlg.Semantics.Term.Polynomial (TFunctor(..))
import UniAlg.Semantics.Category        (tApply)
import UniAlg.Semantics.Schemes
  ( TElim(..)
  , Elim
  , elimToAlg
  , TCoElim(..)
  , CoElim
  , coElimToTerm
  )

import UniAlg.Core.Reduce (reduceTerm)


-- ── Fixed points and Haskell-level recursion schemes ─────────────────────────

-- | Least fixed point of a functor.
newtype Fix f = Fix
  { unFix :: f (Fix f)
  }


-- | Catamorphism (fold) over a 'Fix'-structured value.
cata :: Functor f => (f a -> a) -> Fix f -> a
cata alg =
  alg . fmap (cata alg) . unFix


-- | Anamorphism (unfold) producing a 'Fix'-structured value.
ana :: Functor f => (a -> f a) -> a -> Fix f
ana coalg =
  Fix . fmap (ana coalg) . coalg


-- | Hylomorphism: unfold then fold, fusing coalgebra and algebra in one pass.
hylo :: Functor f => (f b -> b) -> (a -> f a) -> a -> b
hylo alg coalg =
  alg . fmap (hylo alg coalg) . coalg


-- ── TTerm-level recursion schemes for code generation ────────────────────────

-- | Bind the recursive self-reference before running a TTerm recursion scheme.
withSelf :: TTerm a -> ((?self :: TTerm a) => r) -> r
withSelf s k = let ?self = s in k


-- | TTerm-level catamorphism.
--
-- Requires @?self@ to be in scope — use 'withSelf' or 'recModule'.
cataT :: forall f a. (TFunctor f, ?self :: TTerm a)
      => (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
cataT alg x =
  applyAlg @f (\layer -> alg (fmap step layer)) x
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


-- | TTerm-level anamorphism.  Defined as @'hyloT' coalg ('foldToTerm' \@f)@.
--
-- Requires @?self@ to be in scope — use 'withSelf' or 'recModule'.
anaT :: forall f a. (TFunctor f, ?self :: TTerm a)
     => (TTerm a -> TTerm a)
     -> TTerm a -> TTerm a
anaT coalg = hyloT @f coalg (foldToTerm @f)


-- | TTerm-level hylomorphism.  @coalg = id@ collapses to 'cataT'.
--
-- Requires @?self@ to be in scope — use 'withSelf' or 'recModule'.
hyloT :: forall f a. (TFunctor f, ?self :: TTerm a)
      => (TTerm a -> TTerm a)        -- ^ coalgebra
      -> (f (TTerm a) -> TTerm a)    -- ^ algebra
      -> TTerm a -> TTerm a
hyloT coalg alg x =
  applyAlg @f (\layer -> alg (fmap step layer)) (coalg x)
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


-- ── Module constructors ───────────────────────────────────────────────────────

-- Builds forall _a0 .. _a(n-1). _a0 -> .. -> _a(n-1).
-- Attaches an explicit polymorphic type scheme to recursive definitions so
-- Hydra skips inference and avoids the occurs-check on equi-recursive bodies.
polyFnScheme :: Int -> TypeScheme
polyFnScheme n = TypeScheme
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
                 , functionTypeCodomain = b }


-- | Build a 'TermDefinition' for a hylomorphism with shared outer parameters.
--
-- @outerArgNames@ are bound as 'TTerm' values and passed to @k@.  @k@ returns
-- the coalgebra and algebra together, so each name is declared exactly once.
-- Pass @id@ as the coalgebra for a pure catamorphism.  'withSelf' and the
-- partial self-application are handled automatically.
--
-- @
-- recDef \@(SeqF Tensor) \"seed.seq\" \"fold_seq\" [\"w\", \"s0\"] $ \\[w, s0] ->
--   ( id
--   , \\a s -> add (contraction real \"hi,i->h\" w a) s )
-- @
recDef :: forall f a. TFunctor f
       => String
       -> String
       -> [String]
       -> ([TTerm a] -> ( TTerm a -> TTerm a        -- ^ coalgebra
                        , f (TTerm a) -> TTerm a ))  -- ^ algebra
       -> TermDefinition
recDef ns name outerArgNames k = TermDefinition
  { termDefinitionName       = Name (ns <> "." <> name)
  , termDefinitionTerm       = reduceTerm $ unTTerm $ foldr (~>) innerTerm outerArgNames
  , termDefinitionTypeScheme = Just (polyFnScheme (length outerArgNames + 2))
  }
  where
    vars        = map var outerArgNames
    (coalg,alg) = k vars
    appliedSelf = foldl (\s n -> tApply s (var n)) (var (ns <> "." <> name)) outerArgNames
    innerTerm   = "x" ~> withSelf appliedSelf (hyloT @f coalg alg (var "x"))


-- | Wrap a single 'recDef' in a 'Module'.  Primary builder for recursive
-- architectures: declares functor @f@, coalgebra, and algebra together.
--
-- @
-- recModule \@(SeqF Tensor) \"transformer\" \"stack\"
--           [Namespace \"numpy\"] [\"x\", \"tokens\"] $ \\[x, tokens] ->
--   ( id, stackAlg x tokens )
-- @
recModule :: forall f a. TFunctor f
          => String
          -> String
          -> [Namespace]
          -> [String]
          -> ([TTerm a] -> ( TTerm a -> TTerm a
                           , f (TTerm a) -> TTerm a ))
          -> Module
recModule ns name deps outerArgNames k = Module
  { moduleDescription      = Just "Recursive definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm (recDef @f ns name outerArgNames k)]
  }


-- | Pure catamorphism builder.  Algebra expressed as a natural 'Elim'-typed
-- handler — no @Data.Functor.*@ constructors visible to the caller.
--
-- @
-- cataModule \@(SeqF Tensor) \"seed.seq\" \"fold_seq\"
--            [Namespace \"numpy\"] [\"wIn\", \"wRec\", \"b\", \"s0\"] $ \\[wIn, wRec, b, s0] ->
--   ( s0
--   , \\a s -> add (contraction real \"hi,i->h\" wIn a) s
--   )
-- @
cataModule :: forall f a. (TFunctor f, TElim f)
           => String
           -> String
           -> [Namespace]
           -> [String]
           -> ([TTerm a] -> Elim f a (TTerm a))
           -> Module
cataModule ns name deps outerArgNames k =
  recModule @f ns name deps outerArgNames $ \vs -> (id, elimToAlg @f (k vs))


-- | Pure anamorphism builder.  Coalgebra expressed as a natural 'CoElim'-typed
-- function — arch authors produce plain tuples or 'Either's.
--
-- @
-- anaModule \@(StreamF Tensor) \"seed.stream\" \"unfold_stream\"
--           [Namespace \"numpy\"] [] $ \\[] ->
--   \\s -> (s, s)
-- @
anaModule :: forall f a. (TFunctor f, TCoElim f)
          => String
          -> String
          -> [Namespace]
          -> [String]
          -> ([TTerm a] -> (TTerm a -> CoElim f a))
          -> Module
anaModule ns name deps outerArgNames k =
  recModule @f ns name deps outerArgNames $ \vs ->
    (coElimToTerm @f . k vs, foldToTerm @f)
