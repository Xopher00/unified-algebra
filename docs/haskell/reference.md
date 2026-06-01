# Reference

## Known open issues

**Composite semiring operations** — `Semiring` holds string op keys that must
be registered in `backends/*.json`.  This covers all atomic semirings (real,
tropical, log-sum-exp, boolean) but not composite reduction forms such as
`-logsumexp(-x)` (soft-min), where the ⊕ is a wrapped composition rather than
a single backend primitive.  The fix requires changing `Semiring` to hold
`TTerm` functions and adding a combinator layer to keep axis-index internals
out of user code — a non-trivial redesign deferred until an architecture that
needs it is ready to build.

## Module map

```haskell
src/UniAlg.hs                      Top-level re-export surface
src/UniAlg/
  Architecture.hs                  cataModule / anaModule / hyloModule; Elim / CoElim
  Scheme.hs                        Re-export surface for recursion schemes
  Scheme/Internal.hs               cataT / anaT / hyloT / withSelf / Fix
  Term.hs                          TArr, reify, structural morphisms (pair, either, …)
  Term/Internal.hs                 Low-level TTerm construction helpers
  Shape.hs                         Polynomial functor atoms and derived type aliases
  Shape/Encode.hs                  Shape class (matchLayer / buildLayer) and instances
  Tensor.hs                        Semiring, contraction, Einstein notation
  Optics.hs                        Van Laarhoven optics over TTerm values
  Backend.hs                       Backend loading re-export surface
  Backend/Spec.hs                  BackendOp, BackendBinding, LoadedBackend
  Backend/Lowering.hs              Hydra IR rewriting: symbolic names → backend paths
  Backend/Externals.hs             Universe-only backend stub declarations
  Core/BackendSpec.hs              JSON deserialisation types for backend specs
  Core/Ops.hs                      Op resolution (symbolic key → TTerm)
  Core/Ops/Generate.hs             TTerm builders for unary / binary / ternary ops
  Core/Reduce.hs                   Hydra IR simplification (beta, pair, either)
  Codegen.hs                       Flat writers: writePythonWithBackend / loadBackendAndWritePython
                                   Recursive writers: writePythonWithBackendRec / loadBackendAndWritePythonRec
                                   Rec variants skip Hydra type inference for recursive modules.
```
