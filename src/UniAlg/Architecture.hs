{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

{-|
Module-level builders for neural recursion schemes.

The three public builders correspond to the three directions a recursive
architecture can face:

  * 'cataModule' — catamorphism (fold).  Input is a fixed recursive structure;
    the architecture consumes it bottom-up.  Algebra direction only.
    Typical use: 'SeqRnn', 'TreeRnn'.

  * 'anaModule' — anamorphism (unfold).  Input is a seed; the architecture
    produces a corecursive structure top-down.  Coalgebra direction only.
    Typical use: 'Moore', 'Mealy', 'StreamRnn'.

  * 'hyloModule' — hylomorphism (unfold then fold).  Input is a seed; the
    coalgebra unfolds it into a structure, the algebra re-folds it.
    Typical use: 'EdgeConv'.

All three builders share the same shape:

@
  builder \@FunctorType
    "namespace"         -- Hydra module namespace, e.g. "seed.seq"
    "functionName"      -- generated Python function name, e.g. "fold_seq"
    [Namespace "torch"] -- backend namespaces imported by generated code
    ["w", "b"]          -- outer parameter names (weights, biases, …)
    $ \\[w, b] -> …     -- lambda receiving those params as TTerm values
@

The functor type application @\@FunctorType@ is required because GHC cannot
infer which 'Shape' instance to use from the algebra alone.
-}
module UniAlg.Architecture
  ( Elim
  , CoElim
  , hyloDef
  , hyloModule
  , cataModule
  , anaModule
  ) where

import Data.Coerce (coerce)
import Data.Functor.Const (Const(..))
import Data.Functor.Identity (Identity(..))
import Data.Functor.Product (Product(..))
import Data.Functor.Sum (Sum(..))
import qualified Data.Kind as Kind

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

import UniAlg.Core.Reduce (reduceTerm)
import UniAlg.Shape.Encode
  ( Exp(..)
  , Shape(..)
  )
import UniAlg.Scheme.Internal
  ( hyloT
  , withSelf
  )
import UniAlg.Term.Internal
  ( tApply
  , tLam
  , tLeft
  , tPair
  , tRight
  , tVar
  )


-- | Type-level eliminator that converts a functor shape into the curried
-- Haskell type expected by a catamorphism algebra.
--
-- Each functor atom contributes one argument:
--
-- @
-- Elim (Const ())        a r = r                   -- no argument (unit leaf)
-- Elim (Const (TTerm k)) a r = TTerm k -> r        -- fixed-value leaf
-- Elim Identity          a r = TTerm a -> r        -- recursive child
-- Elim (Sum f g)         a r = (Elim f a r, Elim g a r)   -- case pair
-- Elim (Product f g)     a r = Elim f a (Elim g a r)      -- curried args
-- Elim (Exp (TTerm i))   a r = (TTerm i -> TTerm a) -> r  -- function arg
-- @
--
-- The algebra argument to 'cataModule' has type @Elim f a (TTerm a)@.
-- For example, with @f = Sum (Const (TTerm a)) Identity@ (list functor):
--
-- @
-- Elim (Sum (Const (TTerm a)) Identity) a (TTerm a)
--   = (Elim (Const (TTerm a)) a (TTerm a), Elim Identity a (TTerm a))
--   = (TTerm a -> TTerm a, TTerm a -> TTerm a)
-- @
type family Elim (f :: Kind.Type -> Kind.Type) (a :: Kind.Type) (r :: Kind.Type) :: Kind.Type where
  Elim (Const ())        a r = r
  Elim (Const (TTerm k)) a r = TTerm k -> r
  Elim Identity          a r = TTerm a -> r
  Elim (Sum f g)         a r = (Elim f a r, Elim g a r)
  Elim (Product f g)     a r = Elim f a (Elim g a r)
  Elim (Exp (TTerm i))   a r = (TTerm i -> TTerm a) -> r

class Shape f => TElim f where
  elimToAlg :: Elim f a r -> f (TTerm a) -> r

instance TElim (Const ()) where
  elimToAlg r (Const ()) = r

instance TElim (Const (TTerm k)) where
  elimToAlg f (Const k) = f k

instance TElim Identity where
  elimToAlg f (Identity x) = f x

instance (TElim f, TElim g) => TElim (Sum f g) where
  elimToAlg (fl, fr) = \case
    InL l -> elimToAlg @f fl l
    InR r -> elimToAlg @g fr r

instance (TElim f, TElim g) => TElim (Product f g) where
  elimToAlg h (Pair fl gr) = elimToAlg @g (elimToAlg @f h fl) gr

instance TElim (Exp (TTerm i)) where
  elimToAlg h (Exp g) = h g

-- | Type-level co-eliminator that converts a functor shape into the Haskell
-- value a coalgebra must produce for 'anaModule'.
--
-- Each atom maps to a natural Haskell carrier:
--
-- @
-- CoElim (Const ())        a = ()
-- CoElim (Const (TTerm k)) a = TTerm k
-- CoElim Identity          a = TTerm a
-- CoElim (Sum f g)         a = Either (CoElim f a) (CoElim g a)
-- CoElim (Product f g)     a = (CoElim f a, CoElim g a)
-- CoElim (Exp (TTerm i))   a = TTerm i -> TTerm a
-- @
--
-- The coalgebra argument to 'anaModule' has type @TTerm a -> CoElim f a@.
-- For Moore machines with @f = Product (Const (TTerm o)) (Exp (TTerm i))@:
--
-- @
-- CoElim (Product (Const (TTerm o)) (Exp (TTerm i))) a
--   = (TTerm o, TTerm i -> TTerm a)
-- @
type family CoElim (f :: Kind.Type -> Kind.Type) (a :: Kind.Type) :: Kind.Type where
  CoElim (Const ())        a = ()
  CoElim (Const (TTerm k)) a = TTerm k
  CoElim Identity          a = TTerm a
  CoElim (Sum f g)         a = Either (CoElim f a) (CoElim g a)
  CoElim (Product f g)     a = (CoElim f a, CoElim g a)
  CoElim (Exp (TTerm i))   a = TTerm i -> TTerm a
  CoElim (Const (TTerm i -> TTerm o)) a = TTerm i -> TTerm o

class Shape f => TCoElim f where
  coElimToTerm :: CoElim f a -> TTerm a

instance TCoElim (Const ()) where
  coElimToTerm () = TTerm Terms.unit

instance TCoElim (Const (TTerm k)) where
  coElimToTerm k = coerce k

instance TCoElim Identity where
  coElimToTerm x = x

instance (TCoElim f, TCoElim g) => TCoElim (Sum f g) where
  coElimToTerm (Left fv) = tLeft (coElimToTerm @f fv)
  coElimToTerm (Right gv) = tRight (coElimToTerm @g gv)

instance (TCoElim f, TCoElim g) => TCoElim (Product f g) where
  coElimToTerm (fv, gv) = tPair (coElimToTerm @f fv) (coElimToTerm @g gv)

instance TCoElim (Exp (TTerm i)) where
  coElimToTerm fn = tLam "inp" (fn (tVar "inp"))

instance TCoElim (Const (TTerm i -> TTerm o)) where
  coElimToTerm fn = coerce (tLam "inp" (fn (tVar "inp")))

polyFnScheme :: Int -> TypeScheme
polyFnScheme n = TypeScheme
  { typeSchemeVariables = vars
  , typeSchemeBody = foldr step ret (init tvars)
  , typeSchemeConstraints = Nothing
  }
  where
    vars = [Name ("_a" <> show i) | i <- [0 .. n - 1]]
    tvars = fmap TypeVariable vars
    ret = last tvars
    step a b = TypeFunction FunctionType
      { functionTypeDomain = a
      , functionTypeCodomain = b
      }

-- | Build a single 'TermDefinition' for a hylomorphism.
--
-- Lower-level than 'hyloModule'; use this when you need to combine multiple
-- definitions in one module.  'hyloModule' wraps a single 'hyloDef' call.
--
-- The lambda receives the outer parameters as 'TTerm' values and must return
-- a pair @(coalg, alg)@:
--
--   * @coalg :: TTerm a -> TTerm a@ — the unfolding step.  Given the current
--     seed value, produces a term in the range of the functor's 'matchLayer'.
--     For a pure catamorphism pass @id@; for a hylomorphism this typically
--     applies the right adjoint of the forward operation (e.g. @subtract@).
--
--   * @alg :: f (TTerm a) -> TTerm a@ — the folding step.  Receives a Haskell
--     functor value whose recursive positions have already been replaced by
--     self-call terms.  Pattern match on the functor constructors.
hyloDef :: forall f a. Shape f
        => String
        -> String
        -> [String]
        -> ([TTerm a] -> ( TTerm a -> TTerm a
                         , f (TTerm a) -> TTerm a ))
        -> TermDefinition
hyloDef ns name outerArgNames k = TermDefinition
  { termDefinitionName = Name (ns <> "." <> name)
  , termDefinitionTerm = reduceTerm $ unTTerm $ foldr (~>) innerTerm outerArgNames
  , termDefinitionTypeScheme = Just (polyFnScheme (length outerArgNames + 2))
  }
  where
    vars = map var outerArgNames
    (coalg, alg) = k vars
    appliedSelf = foldl (\s n -> tApply s (var n)) (var (ns <> "." <> name)) outerArgNames
    innerTerm = "x" ~> withSelf appliedSelf (hyloT @f coalg alg (var "x"))

-- | Build a 'Module' for a hylomorphism over functor @f@.
--
-- A hylomorphism unfolds a seed value into a structure via the coalgebra,
-- then folds that structure into a result via the algebra.  For a pure
-- catamorphism pass @id@ as the coalgebra; 'cataModule' does this automatically.
--
-- The type application @\@f@ is required:
--
-- @
-- hyloModule \@(EdgeF Tensor)
--   "seed.edge" "edge_conv"
--   [Namespace "torch"] ["w"] $ \\[w] ->
--     ( \\x   -> subtract (second x) (first x)   -- coalgebra: unfold pairs
--     , \\case
--         InL d      -> relu (contraction real "ij,j->i" w d)
--         InR (Pair l r) -> maximum l r          -- algebra: fold differences
--     )
-- @
--
-- Arguments:
--
--   [@ns@]             Hydra module namespace (e.g. @"seed.edge"@).
--   [@name@]           Generated Python function name (e.g. @"edge_conv"@).
--   [@deps@]           Backend namespaces imported by the generated module.
--   [@outerArgNames@]  Names of outer parameters (weights, biases, …).
--   [@k@]             Lambda receiving those parameters as @TTerm@ values;
--                      must return @(coalg, alg)@.
hyloModule :: forall f a. Shape f
           => String
           -> String
           -> [Namespace]
           -> [String]
           -> ([TTerm a] -> ( TTerm a -> TTerm a
                            , f (TTerm a) -> TTerm a ))
           -> Module
hyloModule ns name deps outerArgNames k = Module
  { moduleDescription = Just "Hylomorphic recursive definition"
  , moduleNamespace = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions = [DefinitionTerm (hyloDef @f ns name outerArgNames k)]
  }

recModule :: forall f a. Shape f
          => String
          -> String
          -> [Namespace]
          -> [String]
          -> ([TTerm a] -> ( TTerm a -> TTerm a
                           , f (TTerm a) -> TTerm a ))
          -> Module
recModule = hyloModule

-- | Build a 'Module' for a catamorphism (fold) over functor @f@.
--
-- The algebra argument has type @Elim f a (TTerm a)@, which GHC expands to a
-- curried Haskell function whose argument structure mirrors the functor.
-- For example, with @f = Sum (Const (TTerm Tensor)) Identity@:
--
-- @
-- Elim f Tensor (TTerm Tensor)
--   = (TTerm Tensor -> TTerm Tensor,   -- InL branch: leaf value
--      TTerm Tensor -> TTerm Tensor)   -- InR branch: recursive child
-- @
--
-- The type application @\@f@ is required:
--
-- @
-- cataModule \@(ListF Tensor)
--   "seed.seq" "fold_seq"
--   [Namespace "torch"] ["w"] $ \\[w] ->
--     ( \\()     -> s0           -- InL: empty list
--     , \\a s    -> step w a s  -- InR: head and accumulated state
--     )
-- @
cataModule :: forall f a. (Shape f, TElim f)
           => String
           -> String
           -> [Namespace]
           -> [String]
           -> ([TTerm a] -> Elim f a (TTerm a))
           -> Module
cataModule ns name deps outerArgNames k =
  recModule @f ns name deps outerArgNames $ \vs -> (id, elimToAlg @f (k vs))

-- | Build a 'Module' for an anamorphism (unfold) over functor @f@.
--
-- The coalgebra argument has type @TTerm a -> CoElim f a@, which expands to a
-- function whose return type mirrors the functor in plain Haskell values.
-- For Moore machines with @f = Product (Const (TTerm o)) (Exp (TTerm i))@:
--
-- @
-- TTerm State -> CoElim f State
--   = TTerm State -> (TTerm Output, TTerm Input -> TTerm State)
-- @
--
-- The coalgebra returns a Haskell pair (or nested pairs/Eithers) rather than
-- a 'TTerm'; 'anaModule' handles the encoding back to Hydra IR.
--
-- @
-- anaModule \@(MooreF Output Input)
--   "seed.moore" "moore_step"
--   [Namespace "torch"] ["w"] $ \\[w] ->
--     \\s -> ( decode s               -- output projection
--            , \\inp -> step w s inp  -- state transition
--            )
-- @
anaModule :: forall f a. (Shape f, TCoElim f)
          => String
          -> String
          -> [Namespace]
          -> [String]
          -> ([TTerm a] -> (TTerm a -> CoElim f a))
          -> Module
anaModule ns name deps outerArgNames k =
  recModule @f ns name deps outerArgNames $ \vs ->
    (coElimToTerm @f . k vs, buildLayer @f)
