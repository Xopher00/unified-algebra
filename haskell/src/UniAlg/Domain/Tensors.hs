{-# LANGUAGE OverloadedStrings #-}

{-|
Einstein-notation tensor contractions over semirings, compiled to Hydra IR.

Parse an equation string with 'parseEquation', compose equations with
'fuseEquation' or a 'FusionTree', then compile to a 'TTerm' with
'applyEquation' \/ 'applyTree', or to a codegen 'Module' with
'equationModule' \/ 'treeModule'.
-}
module UniAlg.Domain.Tensors
  ( Tensor
  , Index(..)
  , Orientation(..)
  , Semiring(..)
  , Equation(..)
  , FusionTree
  , parseEquation
  , fuseEquation
  , fusionLeaf
  , fusionNode
  , fuseTree
  , contract
  , adjointContract
  , compileEquation
  , applyEquation
  , compileTree
  , applyTree
  , treeModule
  , contractModule
  , equationModule
  ) where

import Data.List (isInfixOf, nub)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set

import Hydra.Kernel
  ( Definition(..)
  , Module(..)
  , Namespace(..)
  )

import Hydra.Phantoms
  ( TTerm(..)
  )

import Hydra.Dsl.Meta.Phantoms
  ( var
  , (@@)
  , definitionInNamespace
  , toDefinition
  )

import UniAlg.Semantics.Term.Arrows
  ( reify2
  )

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Core.Reduce
  ( reduceTerm
  )

import UniAlg.Core.Ops
  ( op
  )

import UniAlg.Semantics.Term.Polynomial
  ( Const(..)
  , Product(..)
  , RoseF
  )

import UniAlg.Semantics.Recursion
  ( Fix(..)
  , cata
  )


-- | Phantom type tag for tensor-typed 'TTerm' values.
data Tensor


-- | A single Einstein index label (a single character, e.g. @\'i\'@, @\'j\'@).
newtype Index = Index Char
  deriving (Eq, Ord, Show)


-- | Whether to use the forward or adjoint semiring operations.
--
-- 'Forward' uses @times@ for element products and @plus@ for reduction.
-- 'Adjoint' uses @adjoint@ for element products and @times@ for reduction
-- (i.e. the dual contraction for backward passes).
data Orientation
  = Forward
  | Adjoint
  deriving (Eq, Show)


-- | An algebraic structure parameterising tensor contractions.
--
-- @
-- real     = Semiring \"add\"     \"multiply\" (Just \"divide\")
-- tropical = Semiring \"minimum\" \"add\"      Nothing
-- @
data Semiring = Semiring
  { semiringPlus    :: String        -- ^ Op key for reduction (e.g. @\"add\"@).
  , semiringTimes   :: String        -- ^ Op key for element products (e.g. @\"multiply\"@).
  , semiringAdjoint :: Maybe String  -- ^ Op key for the adjoint of @times@ (e.g. @\"divide\"@), if any.
  } deriving (Eq, Show)


-- ── Equation ─────────────────────────────────────────────────────────────────

-- | A parsed Einstein notation equation.
data Equation = Equation
  { eqInputs  :: [[Index]]  -- ^ Index set for each input operand.
  , eqOutput  :: [Index]    -- ^ Output index labels.
  , eqReduced :: [Index]    -- ^ Indices that are summed over (contracted).
  } deriving (Eq, Show)


-- | Parse an Einstein notation string into an 'Equation'.
--
-- Returns @'Left' err@ if the string is malformed (missing arrow, duplicate
-- output labels, output labels absent from all inputs).
--
-- @
-- parseEquation \"ij,jk->ik\"  -- matrix multiply: reduces j
-- parseEquation \"i,j->ij\"    -- outer product: no reduction
-- @
parseEquation :: String -> Either String Equation
parseEquation raw = do
  (lhs, rhs) <- splitArrow (filter (/= ' ') raw)
  let inputs     = (fmap . fmap) Index (splitOn ',' lhs)
      output     = fmap Index rhs
      distinctIn = nub (concat inputs)
      inSet      = Set.fromList distinctIn
      outSet     = Set.fromList output
  check (Set.size outSet == length output) "output labels must be unique"
  check (outSet `Set.isSubsetOf` inSet)    "output labels not in any input"
  pure Equation
    { eqInputs  = inputs
    , eqOutput  = output
    , eqReduced = filter (`Set.notMember` outSet) distinctIn
    }


splitArrow :: String -> Either String (String, String)
splitArrow s
  | "->" `isInfixOf` s = Right (a, drop 2 b)
  | otherwise          = Left "equation must contain '->'"
  where (a, b) = break (== '-') s


check :: Bool -> String -> Either String ()
check True  _ = pure ()
check False e = Left e


-- | Fuse an inner equation into one slot of an outer equation.
--
-- Substitutes the output of @inner@ for the @slot@-th input of @outer@,
-- producing a combined equation over all inputs of both.  The inner
-- output labels must exactly match the replaced outer input labels.
--
-- @
-- -- ij,jk->ik  fused with  kl,lm->km  at slot 1 gives  ij,kl,lm->im
-- @
fuseEquation :: Equation -> Int -> Equation -> Either String Equation
fuseEquation outer slot inner = do
  check (slot >= 0 && slot < length (eqInputs outer))
    ("slot " <> show slot <> " out of range for equation with "
      <> show (length (eqInputs outer)) <> " inputs")
  check (eqOutput inner == eqInputs outer !! slot)
    "inner output labels must match the replaced outer input"
  let inputs    = take slot (eqInputs outer)
               ++ eqInputs inner
               ++ drop (slot + 1) (eqInputs outer)
      allInSet  = Set.fromList (concat inputs)
      outLbls   = eqOutput outer
      outLblSet = Set.fromList outLbls
      reduced   = filter (`Set.notMember` outLblSet) $
                    nub (eqReduced inner ++ eqReduced outer)
  pure Equation
    { eqInputs  = inputs
    , eqOutput  = outLbls
    , eqReduced = filter (`Set.member` allInSet) reduced
    }


-- ── Fusion tree ──────────────────────────────────────────────────────────────

-- | A rose tree of 'Equation's to be fused by catamorphism.
--
-- Each node carries a parent 'Equation' and zero or more child sub-trees.
-- 'fuseTree' collapses the tree into a single flat 'Equation' by fusing
-- children into their parent's input slots from deepest to shallowest.
type FusionTree = Fix (RoseF (Const Equation))


-- | A leaf node — a single 'Equation' with no children to fuse.
fusionLeaf :: Equation -> FusionTree
fusionLeaf eq = Fix (Pair (Const eq) [])


-- | An internal node whose children are fused into the parent equation's
-- corresponding input slots (in order, highest slot first for index stability).
fusionNode :: Equation -> [FusionTree] -> FusionTree
fusionNode eq children = Fix (Pair (Const eq) children)


-- | Collapse a 'FusionTree' into a single 'Equation' by catamorphism.
fuseTree :: FusionTree -> Either String Equation
fuseTree = cata algebra
  where
    algebra :: RoseF (Const Equation) (Either String Equation) -> Either String Equation
    algebra (Pair (Const eq) childResults) =
      -- Highest slot first: keeps lower slot indices stable as each fusion
      -- prepends inner inputs at the targeted position.
      foldr step (Right eq) (zip [0..] childResults)

    step :: (Int, Either String Equation) -> Either String Equation -> Either String Equation
    step (slot, child) acc = do
      a <- acc
      c <- child
      fuseEquation a slot c


-- ── Simple contractions (no equation, arity-inferred reduce) ─────────────────

-- | Curried binary contraction over the given semiring.
contract :: Semiring -> TTerm (Tensor -> Tensor -> Tensor)
contract sr =
  reify2 $ \x y ->
    op ("reduce." <> semiringPlus sr) @@ (op (semiringTimes sr) @@ x @@ y)


-- | Adjoint contraction: uses @adjoint@ for element product and @times@
-- for reduction.  Fails at runtime if the semiring has no adjoint.
adjointContract :: Semiring -> TTerm (Tensor -> Tensor -> Tensor)
adjointContract sr =
  case semiringAdjoint sr of
    Nothing  -> error "adjointContract: semiring has no adjoint"
    Just adj -> reify2 $ \x y ->
      op ("reduce." <> semiringTimes sr) @@ (op adj @@ x @@ y)


-- ── Equation-aware contractions ──────────────────────────────────────────────

-- | Compile an 'Equation' to a curried 'TTerm' lambda.
compileEquation :: Orientation -> Semiring -> Equation -> Either String (TTerm a)
compileEquation orientation sr eq = do
  (timesKey, reduceKey) <- case (orientation, semiringAdjoint sr) of
    (Forward, _)        -> pure (semiringTimes sr, "reduce." <> semiringPlus sr)
    (Adjoint, Just adj) -> pure (adj,              "reduce." <> semiringTimes sr)
    (Adjoint, Nothing)  -> Left "Adjoint orientation requires a semiring with an adjoint"
  let target   = eqOutput eq ++ eqReduced eq
      alignOp inp t =
        let existing    = Set.fromList inp
            expanded    = inp ++ filter (`Set.notMember` existing) target
            posOf       = Map.fromList (zip expanded [0..])
            axes        = [length inp .. length expanded - 1]
            permutation = fmap (\v -> Map.findWithDefault 0 v posOf) target
            unsqueezed  = foldl (\acc ax -> op "structural.expand_dims" @@ acc @@
                                   TTerm (Terms.int32 ax)) t axes
        in  op "structural.transpose" @@ unsqueezed @@
              TTerm (Terms.list (fmap Terms.int32 permutation))
      params   = ["t" <> show i | i <- [0 .. length (eqInputs eq) - 1]]
      aligned  = zipWith (\inp p -> alignOp inp (var p)) (eqInputs eq) params
      product_ = foldl1 (\a b -> op timesKey @@ a @@ b) aligned
      body     = foldl (\acc ax -> op reduceKey @@ acc @@ TTerm (Terms.int32 ax))
                       product_ [length (eqOutput eq) .. length target - 1]
  pure (TTerm (Terms.lambdas params (unTTerm body)))


-- | Compile an 'Equation' and immediately apply it to input tensors.
--
-- Equivalent to @'compileEquation'@ followed by partial application of
-- @args@, with 'reduceTerm' applied to beta-reduce the result.
-- This is the main entry point for constructing tensor contraction terms
-- to embed in algebras or pass directly to 'writePythonWithBackend'.
applyEquation :: Orientation -> Semiring -> Equation -> [TTerm Tensor] -> Either String (TTerm Tensor)
applyEquation orientation sr eq args = do
  compiled <- compileEquation orientation sr eq
  pure $ TTerm $ reduceTerm $ foldl Terms.apply (unTTerm compiled) (fmap unTTerm args)


-- ── Module builder ───────────────────────────────────────────────────────────

contractModule :: String -> Semiring -> Module
contractModule defName sr = moduleFromTerm "unialg.tensors" defName [] (contract sr)


-- | Build a codegen-ready 'Module' from a pre-parsed 'Equation'.
equationModule :: String -> String -> [Namespace] -> Semiring -> Equation -> Orientation -> Either String Module
equationModule ns name deps sr eq orient = do
  t <- compileEquation orient sr eq
  pure (moduleFromTerm ns name deps t)


-- | Collapse a 'FusionTree', then compile the resulting flat 'Equation'.
--
-- Fuses child equations automatically via 'fuseTree', then delegates to
-- 'compileEquation'.  Returns @'Left' err@ on any fusion or orientation error.
compileTree :: Orientation -> Semiring -> FusionTree -> Either String (TTerm a)
compileTree orientation sr tree = fuseTree tree >>= compileEquation orientation sr


-- | Collapse a 'FusionTree', compile, and immediately apply to input tensors.
applyTree :: Orientation -> Semiring -> FusionTree -> [TTerm Tensor]
          -> Either String (TTerm Tensor)
applyTree orientation sr tree args =
  fuseTree tree >>= \eq -> applyEquation orientation sr eq args


-- | Collapse a 'FusionTree' and build a codegen-ready 'Module'.
treeModule :: String -> String -> [Namespace] -> Semiring -> FusionTree -> Orientation
           -> Either String Module
treeModule ns name deps sr tree orient =
  fuseTree tree >>= \eq -> equationModule ns name deps sr eq orient


moduleFromTerm :: String -> String -> [Namespace] -> TTerm a -> Module
moduleFromTerm ns name deps t = Module
  { moduleDescription      = Just "Tensor operation"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      =
      [toDefinition (definitionInNamespace (Namespace ns) name t)]
  }


-- ── Utilities ────────────────────────────────────────────────────────────────

splitOn :: Char -> String -> [String]
splitOn _ [] = [""]
splitOn delim s =
  case break (== delim) s of
    (chunk, [])     -> [chunk]
    (chunk, _:rest) -> chunk : splitOn delim rest
