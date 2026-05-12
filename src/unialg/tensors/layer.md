A clean boundary would be:

```text
1. Syntax layer
   Parses tensor/math expressions into a small AST.
   No Hydra. No backend. No Morphism execution.

2. Semantic layer
   Turns AST into typed Morphism compositions.
   Uses identity, pair, compose, section, etc.
   No NumPy/JAX/Torch imports.

3. Backend primitive layer
   Loads JSON specs.
   Registers backend functions as Hydra Primitives.
   Owns codecs, dtype/device policy, arity, backend quirks.

4. Realization/lowering layer
   Converts Morphism syntax to Hydra Terms.
   No parsing. No backend selection.

5. Execution layer
   run(...)
   Applies/reduces Hydra terms with aux_primitives.
```

For tensor operations, the rule should be:

```text
Parser decides structure.
Semantics decides morphism composition.
Backend layer decides implementation.
Realizer emits Hydra.
Runner executes.
```

So avoid these crossings:

```text
parser importing BackendOps          # bad
realize.py importing numpy/jax/torch # bad
morphisms.py knowing codecs          # bad
backend loader constructing parser ASTs # bad
```

The tensor parser should only need:

```python
symbol_table = {
    "+": "add",
    "*": "multiply",
    "⊕": "logaddexp",
}
```

Then the semantic lowering receives:

```python
ops["add"]
ops["multiply"]
```

as already-built morphisms.

A good file split:

```text
tensors/math.py
    BackendOps, BackendPrimitive, JSON loading, codecs

tensors/expr.py
    AST nodes: Var, Lit, BinOp, UnaryOp

tensors/parser.py
    tokenize + Pratt/precedence parser

tensors/lower.py
    AST -> Morphism using BackendOps

semantics/morphisms.py
    generic typed morphism constructors only

semantics/realize.py
    MorphismExpr -> Hydra Term only
```

That keeps tensor work additive instead of contaminating the core DSL.
