{-# LANGUAGE OverloadedStrings #-}

{-|
Tensor contraction DSL: Einstein notation, semirings, and codegen.

=== Semirings

A 'Semiring' parameterises a contraction: @plus@ reduces over contracted axes,
@times@ computes element-wise products, and @adjoint@ (optional) inverts
@times@ for backward passes.

Standard examples:

@
real     = Semiring \"add\"     \"multiply\" (Just \"divide\")
tropical = Semiring \"minimum\" \"add\"      Nothing
boolean  = Semiring \"maximum\" \"minimum\"  Nothing
@

=== Equations

An 'Equation' is a parsed Einstein notation string:

@
\"ij,jk->ik\"   -- matrix multiply
\"ij,j->i\"     -- matrix-vector multiply
\"i,j->ij\"     -- outer product
\"ij,ij->\"     -- Frobenius inner product (full trace)
@

'parseEquation' validates the string and returns an 'Equation' recording
the input index sets, output labels, and reduced (contracted) indices.

'fuseEquation' composes two equations by substituting one equation's output
into a slot of another — enabling multi-operand contractions like
@ij,jk,kl->il@ to be expressed as two fused binary contractions.

=== Code generation

'applyEquation' is the main entry point: given an orientation, a semiring,
a parsed equation, and a list of input 'TTerm' tensors, it returns a 'TTerm'
representing the contraction.  The generated Hydra IR is lowered to calls
like @numpy.sum(numpy.multiply(...))@ after backend substitution.

'tensorOp', 'equationModule', and 'tensorOpModule' wrap these into 'Module'
builders for use with 'writePythonWithBackend'.

=== Op resolution

Backend ops are resolved through the registry in "UniAlg.Core.Ops".
-}
module UniAlg.Domain.Tensors
  ( Tensor
  , Index(..)
  , Orientation(..)
  , Semiring(..)
  , Equation(..)
  , AlignmentPlan(..)
  , FusionTree
  , parseEquation
  , fuseEquation
  , fusionLeaf
  , fusionNode
  , fuseTree
  , alignmentPlan
  , alignmentPlans
  , reducedAxes
  , targetVars
  , contract
  , adjointContract
  , compileEquation
  , applyEquation
  , contractModule
  , tensorOp
  , equationModule
  , tensorOpModule
  ) where

import Data.List (elemIndex, isInfixOf, nub)
import Data.Maybe (fromMaybe)

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

import UniAlg.Semantics.Arrows
  ( reify2
  )

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Core.Reduce
  ( reduceTerm
  )

import UniAlg.Core.Ops
  ( op
  )

import UniAlg.Semantics.Functors
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


-- | The unsqueeze and transpose plan for aligning one input operand to the
-- shared broadcast shape before element-wise multiplication.
data AlignmentPlan = AlignmentPlan
  { unsqueezeAxes :: [Int]  -- ^ Axes to insert via @expand_dims@.
  , perm          :: [Int]  -- ^ Transpose permutation to apply after unsqueezing.
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
  let inputs = (fmap . fmap) Index (splitOn ',' lhs)
      output = fmap Index rhs
      allIn  = nub (concat inputs)
  check (nub output == output) "output labels must be unique"
  check (all (`elem` allIn) output) "output labels not in any input"
  pure Equation
    { eqInputs  = inputs
    , eqOutput  = output
    , eqReduced = filter (`notElem` output) allIn
    }


splitArrow :: String -> Either String (String, String)
splitArrow s
  | "->" `isInfixOf` s = Right (a, drop 2 b)
  | otherwise          = Left "equation must contain '->'"
  where (a, b) = break (== '-') s


check :: Bool -> String -> Either String ()
check True  _ = pure ()
check False e = Left e


targetVars :: Equation -> [Index]
targetVars eq = eqOutput eq ++ eqReduced eq


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
  let inputs  = take slot (eqInputs outer)
             ++ eqInputs inner
             ++ drop (slot + 1) (eqInputs outer)
      allIn   = nub (concat inputs)
      outSet  = eqOutput outer
      reduced = filter (`notElem` outSet) $
                  nub (eqReduced inner ++ eqReduced outer)
  pure Equation
    { eqInputs  = inputs
    , eqOutput  = outSet
    , eqReduced = filter (`elem` allIn) reduced
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
      foldr step (Right eq) (reverse (zip [0..] childResults))

    step :: (Int, Either String Equation) -> Either String Equation -> Either String Equation
    step (slot, child) acc = do
      a <- acc
      c <- child
      fuseEquation a slot c


alignmentPlan :: Equation -> Int -> AlignmentPlan
alignmentPlan eq i = AlignmentPlan
  { unsqueezeAxes = [length inp .. length expanded - 1]
  , perm          = fmap (\v -> fromMaybe 0 (elemIndex v expanded)) target
  }
  where
    inp      = eqInputs eq !! i
    target   = targetVars eq
    existing = nub inp
    expanded = inp ++ filter (`notElem` existing) target


alignmentPlans :: Equation -> [AlignmentPlan]
alignmentPlans eq = fmap (alignmentPlan eq) [0 .. length (eqInputs eq) - 1]


reducedAxes :: Equation -> [Int]
reducedAxes eq =
  [length (eqOutput eq) .. length (targetVars eq) - 1]



-- ── Simple contractions (no equation, arity-inferred reduce) ─────────────────

-- | Curried binary contraction over the given semiring.
--
-- @contract sr@ produces @plus(times(x, y))@ — element product then
-- full reduction.  No alignment or permutation; inputs must have the same
-- shape.
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

align :: AlignmentPlan -> TTerm a -> TTerm a
align plan t = transpose_ (unsqueeze_ t)
  where
    unsqueeze_ t' =
      foldl (\acc ax -> op "structural.expand_dims" @@ acc @@ TTerm (Terms.int32 ax))
            t' (unsqueezeAxes plan)
    transpose_ t' =
      op "structural.transpose" @@ t' @@ TTerm (Terms.list (fmap Terms.int32 (perm plan)))


orientedOps :: Orientation -> Semiring -> Either String (String, String)
orientedOps Forward sr = pure (semiringTimes sr, "reduce." <> semiringPlus sr)
orientedOps Adjoint sr =
  case semiringAdjoint sr of
    Nothing  -> Left "Adjoint orientation requires a semiring with an adjoint"
    Just adj -> pure (adj, "reduce." <> semiringTimes sr)


-- | Compile an 'Equation' to a curried 'TTerm' function.
--
-- Produces a lambda over all input operands that aligns each input (via
-- @expand_dims@ and @transpose@), computes element-wise products, and
-- reduces over the contracted axes.  The result is still Hydra IR using
-- symbolic backend names; lowering resolves those to backend paths.
compileEquation :: Orientation -> Semiring -> Equation -> Either String (TTerm a)
compileEquation orientation sr eq = do
  (timesKey, reduceKey) <- orientedOps orientation sr
  let params   = ["t" <> show i | i <- [0 .. length (eqInputs eq) - 1]]
      aligned  = zipWith (\plan p -> align plan (var p)) (alignmentPlans eq) params
      product_ = foldl1 (\a b -> op timesKey @@ a @@ b) aligned
      body     = foldl (\acc ax -> op reduceKey @@ acc @@ TTerm (Terms.int32 ax))
                       product_ (reducedAxes eq)
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


-- | Parse an equation string and compile it to a composable 'TTerm'.
--
-- The result can be applied with @('@@')@, passed as a leaf into
-- recursion-scheme algebras, or composed with other 'TTerm' functions.
-- Returns @'Left' err@ on a parse or orientation error.
tensorOp :: Semiring -> String -> Orientation -> Either String (TTerm a)
tensorOp sr eqStr orient = do
  eq <- parseEquation eqStr
  compileEquation orient sr eq


-- | Build a codegen-ready 'Module' from a pre-parsed 'Equation'.
equationModule :: String -> String -> [Namespace] -> Semiring -> Equation -> Orientation -> Either String Module
equationModule ns name deps sr eq orient = do
  t <- compileEquation orient sr eq
  pure (moduleFromTerm ns name deps t)


-- | Parse an equation string and build a codegen-ready 'Module'.
--
-- Convenience wrapper combining 'parseEquation', 'compileEquation', and
-- 'moduleFromTerm'.
tensorOpModule :: String -> String -> [Namespace] -> Semiring -> String -> Orientation -> Either String Module
tensorOpModule ns name deps sr eqStr orient = do
  t <- tensorOp sr eqStr orient
  pure (moduleFromTerm ns name deps t)


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
