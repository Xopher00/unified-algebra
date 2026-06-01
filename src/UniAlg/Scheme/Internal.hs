{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

{-|
Plain and code-generating recursion schemes.

The plain schemes ('cata', 'ana', 'hylo') operate on real Haskell values
structured as 'Fix' points.  They are used to build the input trees that
get passed to the @TTerm@-level schemes.

The @TTerm@-level schemes ('cataT', 'anaT', 'hyloT') emit Hydra IR rather
than reducing to a Haskell value.  Each recursive step emits a self-application
of the generated Python function, threading shared outer parameters (weights,
biases) automatically via the implicit parameter '?self'.

'withSelf' binds '?self' once at the definition site.  Every recursive call
inside a @cataT@\/'hyloT@\/'anaT@ body then uses that binding to emit the
correct Python self-call without any explicit threading in user code.

Typical call path inside a 'cataModule' body:

@
withSelf self $ \\() ->
  cataT @f alg input
  -- inside cataT: each recursive child emits
  --   TTerm (apply (unTTerm ?self) (unTTerm child))
  -- which generates: fold_fn(w, child)  in Python
@
-}
module UniAlg.Scheme.Internal
  ( Fix(..)
  , cata
  , ana
  , hylo
  , withSelf
  , cataT
  , anaT
  , hyloT
  ) where

import Hydra.Phantoms (TTerm(..), unTTerm)
import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Shape.Encode
  ( Shape(..)
  )


-- | Fixed point of a functor; used to build input trees for the plain schemes.
newtype Fix f = Fix
  { unFix :: f (Fix f)
  }

-- | Standard catamorphism: fold a 'Fix' tree bottom-up with @alg@.
cata :: Functor f => (f a -> a) -> Fix f -> a
cata alg =
  alg . fmap (cata alg) . unFix

-- | Standard anamorphism: unfold a seed top-down with @coalg@.
ana :: Functor f => (a -> f a) -> a -> Fix f
ana coalg =
  Fix . fmap (ana coalg) . coalg

-- | Standard hylomorphism: unfold with @coalg@, then fold with @alg@.
hylo :: Functor f => (f b -> b) -> (a -> f a) -> a -> b
hylo alg coalg =
  alg . fmap (hylo alg coalg) . coalg

-- | Bind the implicit '?self' parameter used by 'cataT', 'anaT', and 'hyloT'.
--
-- '?self' is the partially-applied @TTerm@ for the recursive function being
-- defined (e.g. @fold_seq(w, s0)@).  Binding it once here means every
-- recursive step inside the body can emit the correct Python self-call
-- without any explicit threading.
withSelf :: TTerm a -> ((?self :: TTerm a) => r) -> r
withSelf s k = let ?self = s in k

-- | Code-generating catamorphism over functor @f@.
--
-- 'matchLayer' pattern-matches the input term against @f@; each recursive
-- position is replaced by a self-call @?self(child)@; the algebra then
-- receives the processed functor layer and produces the output term.
cataT :: forall f a. (Shape f, ?self :: TTerm a)
      => (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
cataT alg x =
  matchLayer @f (\layer -> alg (fmap step layer)) x
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))

-- | Code-generating anamorphism over functor @f@.
--
-- Implemented as a hylomorphism with 'buildLayer' as the algebra — the
-- coalgebra unfolds the seed and 'buildLayer' re-encodes the Haskell functor
-- value back into a @TTerm@.
anaT :: forall f a. (Shape f, ?self :: TTerm a)
     => (TTerm a -> TTerm a)
     -> TTerm a -> TTerm a
anaT coalg = hyloT @f coalg (buildLayer @f)

-- | Code-generating hylomorphism over functor @f@.
--
-- Applies @coalg@ to the input, pattern-matches the result against @f@
-- via 'matchLayer', replaces recursive positions with self-calls, then
-- applies @alg@.  Combining @coalg = id@ recovers 'cataT'.
hyloT :: forall f a. (Shape f, ?self :: TTerm a)
      => (TTerm a -> TTerm a)
      -> (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
hyloT coalg alg x =
  matchLayer @f (\layer -> alg (fmap step layer)) (coalg x)
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))
