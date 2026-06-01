# Generating Code

Use this path when you want to compose backend primitives and emit code without
building a recursive architecture.

## Build a term

Backend ops are ordinary Haskell bindings over `TTerm` values. They come from
the keys declared in `backends/*.json`.

```haskell
import UniAlg

transform :: TTerm Tensor -> TTerm Tensor
transform x =
  neg (add x x)
```

The expression above builds symbolic Hydra IR. It does not execute the backend
operation in Haskell.

## Compose morphisms

Use `TArr` when the composition itself is the object you want to name, pass
around, or reuse.

```haskell
import UniAlg

double :: TArr Tensor Tensor
double = TArr (\x -> add x x)

flipSign :: TArr Tensor Tensor
flipSign = TArr neg

transform :: TTerm Tensor -> TTerm Tensor
transform = runTArr (double >>> flipSign)
```

The composition `double >>> flipSign` is a morphism. `runTArr` applies that
morphism to a `TTerm` input and produces another `TTerm`.

## Generate a module

Use `generatePython` for a flat list of named terms. No `SeedEntry`, functor
shape, or recursion direction is required.

```haskell
import UniAlg

transform :: TTerm Tensor -> TTerm Tensor
transform x =
  neg (add x x)

main :: IO ()
main = generatePython
  "generated/torch"
  "backends/torch.json"
  "transforms"
  [("transform", reify transform)]
```

`reify` turns a Haskell function over `TTerm` values into a lambda term that can
be emitted as code.

## Use a custom backend

The same term can target a different domain by changing the backend JSON file.
The generated term still references symbolic op keys such as `add` and `neg`;
lowering rewrites them to the paths declared by the selected backend.

```haskell
main :: IO ()
main = generatePython
  "generated/mylib"
  "backends/mylib.json"
  "demo"
  [("transform", reify transform)]
```

The backend must declare every op key used by the term.
