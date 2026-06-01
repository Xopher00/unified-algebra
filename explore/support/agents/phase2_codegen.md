# Phase 2 — Codegen + Reference Adapter

## Role

You receive a spec file produced by Phase 1. You:

1. Fill the `ref` block with the strongest available framework reference.
2. Run the renderers to generate the Haskell seed and Python harness.
3. Ensure the build compiles before handing off to Phase 3.

---

## Hard boundary

The `arch` and `cell` blocks come from Phase 1 and are **read-only**.
Do not edit them for any reason. They are composed entirely of PolyF grammar
atoms and explicit `Contraction` / `ElemOp` / `Activation` primitives.
The Haskell seed is always rendered from those primitives.

If you believe a `cell` binding is wrong, stop and flag it to the human
rather than correcting it yourself.

---

## Reference-source cascade

For the `ref` block, pick the strongest source available:

**1. Single framework class** (`ref.strength = "single-class"`)

Introspect `torch.nn` and `tf.keras.layers` for a class whose constructor
arguments and forward semantics match the `cell` bindings:

- Use `inspect.signature`, `__doc__`, `dir()` — never `inspect.getsource`.
- Match means: same number of distinct weight matrices / bias vectors, same
  activation, same einsum structure.
- Record the full instantiation expression in `ref.torch.class` and
  `ref.tensorflow.class`, including any constructor arguments that fix the
  cell to the right shape.  Example:
  `"torch.nn.RNNCell(inp_size, hidden, nonlinearity=\"tanh\")"`.
  Include `dtype=tf.float64` in the TF expression.
- Set `ref.torch.kwargs` and `ref.tensorflow.kwargs` to `{}` (all args go
  in the class string).

**2. Composition of framework primitives** (`ref.strength = "composed"`)

When no single class matches, try composing `torch.nn` / `tf.keras.layers`
primitives (e.g. `Linear → tanh`, `Conv2d + LayerNorm`).

Record the composition as `ref.torch = {"tag": "Composed", "components": [...]}`.
The `components` list should be a sequence of instantiation expressions in
execution order.

**3. Numpy-only** (`ref.strength = "numpy-only"`)

When no framework composition fits (exotic semiring, unusual functor), set:

```json
"ref": {
  "strength": "numpy-only",
  "torch": null,
  "tensorflow": null,
  "numpy": {"tag": "NumpyOnly", "equation": "<brief description>"}
}
```

The numpy reference is auto-generated from the ANF bindings by `render_py.py`
and does not need to be hand-written.

---

## Framework weight conventions

When the `ref` block contains a `SingleClass` entry, the `arch.py` test stub
will need weights copied from the spec parameters into the framework cell.
Record the convention in a comment in `arch.py` so Phase 3 (or the human)
can complete it:

- **Torch**: weights are stored as `weight_ih` (shape `[hidden, inp]`) and
  `weight_hh` (shape `[hidden, hidden]`), `bias_ih`, `bias_hh`.
  Copy directly (no transpose needed).
- **TensorFlow**: `kernel` is shape `[inp, hidden]` (transposed relative to
  the parameter), `recurrent_kernel` is `[hidden, hidden]` (transposed),
  `bias` is `[hidden]`.
  Copy with `.numpy().T` or `tf.transpose`.

---

## Running the renderers

After filling the `ref` block, run both renderers in sequence:

```bash
# Haskell seed
dist-newstyle/build/x86_64-linux/ghc-9.10.2/unialg-0.1.0.0/x/explore-render/build/explore-render/explore-render \
  explore/support/template/specs/<label>.json

# Regenerate Catalogue.hs
runghc explore/gen-catalogue.hs

# Python harness
UV_CACHE_DIR=$TMPDIR/uv-cache uv run python \
  explore/support/template/render_py.py \
  explore/support/template/specs/<label>.json
```

Then build to confirm the generated Haskell compiles:

```bash
cabal test explore-test
```

`cabal test explore-test` also generates the Hydra Python modules under
`explore/archs/<label>/generated/`. This must complete without errors before
handoff to Phase 3.

---

## Output

- Filled spec file at `explore/support/template/specs/<label>.json`
  (the `arch` and `cell` blocks unchanged; `ref` block populated).
- Generated `explore/archs/<label>/ElmanRnn.hs` (or equivalent module name).
- Generated `explore/archs/<label>/arch.py` and `__init__.py`.
- Generated `explore/archs/<label>/generated/<backend>/seed/<label>.py`
  for each backend (produced by `cabal test explore-test`).

If `cabal test explore-test` fails at the compilation stage (not pytest),
diagnose the error and fix the spec or the renderer. Do not proceed to Phase 3
with a compilation failure.
