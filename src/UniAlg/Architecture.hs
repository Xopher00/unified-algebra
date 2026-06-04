{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
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
  , cataDef
  , cataModule
  , anaDef
  , anaModule
  , hyloDef
  , hyloModule
  , polyFnScheme
  , typedLambdas
  , hyloDefWith
  , plainDefWith
  ) where

import Data.Functor.Const (Const(..))
import Data.Functor.Identity (Identity(..))
import Data.Functor.Product (Product(..))
import Data.Functor.Sum (Sum(..))
import qualified Data.Kind as Kind

import Hydra.Phantoms (TTerm(..), unTTerm)
import qualified Hydra.Dsl.Terms as Terms
import Hydra.Dsl.Meta.Phantoms
  ( var )
import Hydra.Kernel
  ( Definition(..)
  , FunctionType(..)
  , Module(..)
  , Name(..)
  , Namespace(..)
  , Term
  , TermDefinition(..)
  , Type(..)
  , TypeScheme(..)
  )

import UniAlg.Reduce (reduceTerm)
import UniAlg.Shape.Encode
  ( Exp(..)
  , Shape(..)
  , CoElim
  , TCoElim(..)
  )
import UniAlg.Scheme.Internal
  ( cataT
  , anaT
  , hyloT
  , withSelf
  )
import UniAlg.Term.Internal
  ( tApply )


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

polyFnScheme :: Int -> TypeScheme
polyFnScheme n = TypeScheme
  { typeSchemeVariables = vars
  , typeSchemeBody = foldr step ret (init tvars)
  , typeSchemeConstraints = Nothing
  }
  where
    vars = [Name ("a" <> show i) | i <- [0 .. n - 1]]
    tvars = fmap TypeVariable vars
    ret = last tvars
    step a b = TypeFunction FunctionType
      { functionTypeDomain = a
      , functionTypeCodomain = b
      }

-- | Build a single 'TermDefinition' for a catamorphism.
--
-- Lower-level than 'cataModule'; use this when you need to combine multiple
-- definitions in one module.  'cataModule' wraps a single 'cataDef' call.
--
-- The lambda receives the outer parameters as 'TTerm' values and must return
-- an algebra function:
--
--   * @alg :: f (TTerm a) -> TTerm a@ — the folding step.  Receives a Haskell
--     functor value whose recursive positions have already been replaced by
--     self-call terms.  Pattern match on the functor constructors.
cataDef :: forall f a. (Shape f, TElim f)
        => String
        -> String
        -> [String]
        -> ([TTerm a] -> Elim f a (TTerm a))
        -> TermDefinition
cataDef ns name outerArgNames k =
  hyloDefWith ns name outerArgNames $ \self vs ->
    withSelf self (cataT @f (elimToAlg @f (k vs)) (var "x"))

-- | Build a single 'TermDefinition' for an anamorphism.
--
-- Lower-level than 'anaModule'; use this when you need to combine multiple
-- definitions in one module.  'anaModule' wraps a single 'anaDef' call.
--
-- The lambda receives the outer parameters as 'TTerm' values and must return
-- a coalgebra function:
--
--   * @coalg :: TTerm a -> CoElim f a@ — the unfolding step.  Given the current
--     seed value, produces a 'CoElim'-shaped Haskell value representing one
--     layer of the structure.  The coalgebra outcome is automatically re-encoded
--     into Hydra IR and then reconstructed with 'buildLayer'.
anaDef :: forall f a. (TCoElim f)
       => String
       -> String
       -> [String]
       -> ([TTerm a] -> (TTerm a -> CoElim f a))
       -> TermDefinition
anaDef ns name outerArgNames k =
  hyloDefWith ns name outerArgNames $ \self vs ->
    withSelf self (anaT @f (k vs) (var "x"))

-- | Build a single 'TermDefinition' for a hylomorphism.
--
-- Lower-level than 'hyloModule'; use this when you need to combine multiple
-- definitions in one module.  'hyloModule' wraps a single 'hyloDef' call.
--
-- The lambda receives the outer parameters as 'TTerm' values and must return
-- a pair @(coalg, alg)@:
--
--   * @coalg :: TTerm a -> CoElim f a@ — the unfolding step.  Given the current
--     seed value, produces a 'CoElim'-shaped value in the range of the functor.
--     For a pure catamorphism, this would produce a layer with no recursive
--     positions; for a hylomorphism this typically unfolds the input.
--
--   * @alg :: f (TTerm a) -> TTerm a@ — the folding step.  Receives a Haskell
--     functor value whose recursive positions have already been replaced by
--     self-call terms.  Pattern match on the functor constructors.
hyloDef :: forall f a. (TCoElim f)
        => String
        -> String
        -> [String]
        -> ([TTerm a] -> ( TTerm a -> CoElim f a
                         , f (TTerm a) -> TTerm a ))
        -> TermDefinition
hyloDef ns name outerArgNames k =
  hyloDefWith ns name outerArgNames $ \self vs ->
    let (coalg, alg) = k vs
    in withSelf self (hyloT @f coalg alg (var "x"))

-- | Build a recursive 'TermDefinition' whose body is a chain of typed outer
-- lambdas wrapped around an inner body. The caller supplies @mkInner@,
-- which receives the recursive 'self' term (the partially-applied function
-- bound to the outer args) and the outer-arg 'TTerm' vars, and returns the
-- inner body. Use 'withSelf' inside @mkInner@ when calling recursion
-- helpers that require the @?self@ implicit param.
--
-- 'hyloDef' (this module) and 'UniAlg.RuntimeArchitecture.hyloDefR' both
-- delegate to this builder; they differ only in which recursion helper
-- they call inside @mkInner@.
hyloDefWith
  :: forall a
   . String
  -> String
  -> [String]
  -> (TTerm a -> [TTerm a] -> TTerm a)
  -> TermDefinition
hyloDefWith ns name outerArgNames mkInner = TermDefinition
  { termDefinitionName       = Name (ns <> "." <> name)
  , termDefinitionTerm       =
      reduceTerm $ typedLambdas (outerArgNames ++ ["x"]) (unTTerm bodyTerm)
  , termDefinitionTypeScheme = Just (polyFnScheme (length outerArgNames + 2))
  }
  where
    vars        = map var outerArgNames
    appliedSelf = foldl (\s n -> tApply s (var n))
                        (var (ns <> "." <> name))
                        outerArgNames
    bodyTerm    = mkInner appliedSelf vars

-- | Build a non-recursive 'TermDefinition'. Wraps a body term (built from
-- the outer-arg 'TTerm' vars) in a typed-lambda chain. Used by
-- 'UniAlg.RuntimeArchitecture.moduleR' for plain (non-self-applying)
-- module definitions.
plainDefWith
  :: forall a
   . String
  -> String
  -> [String]
  -> ([TTerm a] -> TTerm a)
  -> TermDefinition
plainDefWith ns name outerArgNames k = TermDefinition
  { termDefinitionName       = Name (ns <> "." <> name)
  , termDefinitionTerm       =
      reduceTerm $ typedLambdas outerArgNames (unTTerm (k vars))
  , termDefinitionTypeScheme = Just (polyFnScheme (length outerArgNames + 1))
  }
  where
    vars = map var outerArgNames

-- | Wrap @body@ in a curried chain of typed lambdas. The @i@-th parameter is
-- annotated with @TypeVariable (Name ("a" <> show i))@, matching the variables
-- introduced by 'polyFnScheme' so Hydra's analyzer can attach concrete
-- parameter-type annotations to the emitted Python @def@.
typedLambdas :: [String] -> Term -> Term
typedLambdas names body =
  foldr (\(i, n) acc ->
          Terms.lambdaTyped n (TypeVariable (Name ("a" <> show i))) acc)
        body
        (zip [0 :: Int ..] names)

-- | Build a 'Module' for a hylomorphism over functor @f@.
--
-- A hylomorphism unfolds a seed value into a structure via the coalgebra,
-- then folds that structure into a result via the algebra.
--
-- The type application @\@f@ is required:
--
-- @
-- hyloModule \@(EdgeF Tensor)
--   "seed.edge" "edge_conv"
--   [Namespace "torch"] ["w"] $ \\[w] ->
--     ( \\x   -> (subtract (second x) (first x))  -- coalgebra: return CoElim
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
hyloModule :: forall f a. (TCoElim f)
           => String
           -> String
           -> [Namespace]
           -> [String]
           -> ([TTerm a] -> ( TTerm a -> CoElim f a
                            , f (TTerm a) -> TTerm a ))
           -> Module
hyloModule ns name deps outerArgNames k = Module
  { moduleDescription = Just "Hylomorphic recursive definition"
  , moduleNamespace = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions = [DefinitionTerm (hyloDef @f ns name outerArgNames k)]
  }

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
cataModule ns name deps outerArgNames k = Module
  { moduleDescription = Just "Catamorphic recursive definition"
  , moduleNamespace = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions = [DefinitionTerm (cataDef @f ns name outerArgNames k)]
  }

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
anaModule :: forall f a. (TCoElim f)
          => String
          -> String
          -> [Namespace]
          -> [String]
          -> ([TTerm a] -> (TTerm a -> CoElim f a))
          -> Module
anaModule ns name deps outerArgNames k = Module
  { moduleDescription = Just "Anamorphic recursive definition"
  , moduleNamespace = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions = [DefinitionTerm (anaDef @f ns name outerArgNames k)]
  }
