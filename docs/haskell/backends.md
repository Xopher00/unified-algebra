# Backends

## Backend JSON specs

Backend specs use JSON:

The repository includes ready-to-use backend specs at `backends/numpy.json`,
`backends/torch.json`, `backends/tensorflow.json`, `backends/jax.json`, and
`backends/cupy.json`.

```json
{
  "backend": "numpy",
  "structural": {
    "expand_dims": "numpy.expand_dims",
    "transpose":   "numpy.transpose"
  },
  "ops": {
    "add":             { "kind": "elementwise_binary", "path": "numpy.add",      "arity": 2 },
    "multiply":        { "kind": "elementwise_binary", "path": "numpy.multiply", "arity": 2 },
    "tanh":            { "kind": "unary",              "path": "numpy.tanh",     "arity": 1 },
    "reduce.add":      { "kind": "reduce",             "path": "numpy.sum",      "arity": 2 },
    "reduce.multiply": { "kind": "reduce",             "path": "numpy.prod",     "arity": 2 }
  }
}
```

At DSL time, morphisms reference ops symbolically as `unialg.backend.add`.
The lowering pass (`UniAlg.Backend.Lowering`) rewrites these names to
`numpy.add` before Hydra generates Python source.

External module stubs (`UniAlg.Backend.Externals`) declare eta-expanded
definitions for every backend op at the correct arity so Hydra's type system
can resolve them during code generation. These stubs appear in the `universe`
list but are not emitted as output.

### Custom backends

A backend JSON file is just a name mapping. You can create one for any domain.
Place the file anywhere and pass its path to `generatePython`, or pass its
directory and backend name to `loadBackendAndWritePython`:

```json
{
  "backend": "mylib",
  "structural": { ... },
  "ops": {
    "add":        { "kind": "elementwise_binary", "path": "mylib.combine", "arity": 2 },
    "multiply":   { "kind": "elementwise_binary", "path": "mylib.times",   "arity": 2 },
    "reduce.add": { "kind": "reduce",             "path": "mylib.fold",    "arity": 2 }
  }
}
```

Op names in the file become the op keys available to generated terms and
`Semiring` values. For semiring contractions, `semiringPlus = "add"` expects a
matching reduction key named `reduce.add`. The backend path can still point to a
domain-specific operation such as `mylib.combine`; the logical key is the stable
name used by Haskell terms.
