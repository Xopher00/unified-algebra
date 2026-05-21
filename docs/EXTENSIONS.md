# Extension Authoring Guide

## Purpose

This document specifies how to extend unialg for a new domain without rewriting it. It is the contract between the core library and any plugin: what a plugin may do, what it must not do, where each piece of code belongs, and what counts as a complete extension.

The guide is domain-agnostic. The same rules apply whether you are extending unialg with tensor algebra, musical composition, legal reasoning, ecological modeling, cognitive architectures, chemical synthesis, or any other domain that has compositional structure. The shape of the extension does not depend on the shape of the domain.

If you are extending unialg, read this document before writing any code.

---

## The core idea

unialg has a fixed substrate of categorical machinery: morphisms, functors, optics, recursion schemes, monads, and a native backend boundary. Every extension elaborates into that substrate. The substrate does not change to accommodate extensions; extensions express themselves in terms of what the substrate already provides.

This is not a limitation. The substrate is general enough to express any compositional domain. What a domain typically needs is a *vocabulary* — a way of writing things that is natural for the domain — and a *strategy* — a way of converting that vocabulary into substrate machinery. An extension provides both, without changing the substrate.

The benefit is that any two extensions compose. A morphism defined by the tensor extension and a morphism defined by the music extension can be composed with `>>`, paired with `&`, lifted through a functor with `F{f}`, folded with `cata[F](alg)`, and so on, because both extensions ultimately produce ordinary `Morphism` objects. Plugins do not need to know about each other.

---

## The extension contract

Every extension must obey the following rules. These are not stylistic preferences. They are what makes the extension system work.

### Rule 1: Every extension elaborates into the existing substrate

An extension may introduce new vocabulary, helpers, parsers, validators, or domain-specific optimizers. But the final output of an extension must be one of the following:

- `Morphism` — a typed arrow that can be composed with other morphisms
- `Functor` — a polynomial functor that can act on objects and morphisms
- `Optic` — a structural boundary that morphisms act through
- A new `Carrier` (recursive type) defined by a functor
- A new monad descriptor

If an extension produces something that is not one of these, it has not elaborated into the substrate and other extensions cannot interoperate with it. The substrate is the lingua franca.

A corollary: an extension must not require changes to the substrate to function. If you find yourself wanting to modify `semantics/morphisms.py`, `semantics/functors.py`, or `semantics/optics.py` to make your extension work, stop and reconsider. Either your extension is doing too much, or it has identified a real gap in the substrate that should be discussed independently of the extension.

### Rule 2: No parallel expression universes

An extension may have its own internal representation — a parsed specification, a normalized form, a domain-specific intermediate language. But that representation must be a *staging area* for producing substrate objects, not a substitute for them.

The substrate's expression language is `MorphismExpr` from `syntax/expressions.py`. This is what semantic construction operates on and what realization translates to Hydra terms. An extension may have a domain-specific spec format that is convenient for users, but at some point that spec must turn into `MorphismExpr` nodes wrapped in `Morphism` objects.

The wrong pattern: domain spec → custom evaluator → direct backend call, bypassing the morphism layer.

The right pattern: domain spec → `Morphism` constructed using existing combinators → realization → backend execution.

### Rule 3: Each layer of the extension owns one concern

unialg's layered architecture — syntax, semantics, structure, runtime — exists because mixing layer concerns produces code that is hard to test, hard to extend, and hard to reason about. Extensions reproduce this layering at smaller scale.

For a typical extension, the layers map as follows:

| Layer | Owns | Imports from substrate |
|-------|------|------------------------|
| Surface notation | Parsing domain-specific text into a pure spec | None (or syntax helpers) |
| Domain semantics | Converting specs into `Morphism` objects | `semantics/`, `objects.py` |
| Domain structure | Custom recursion schemes, optics, or term shapes if needed | `semantics/`, `structure/` |
| Runtime adapters | Domain-specific native primitives if needed | `runtime/` |

Not every extension needs all four layers. A simple extension might be one file. A complex extension might span all four. The principle is that within whatever files exist, each file owns one concern.

### Rule 4: Surface notation parses, it does not interpret

If your extension provides custom notation — an equation syntax, a domain-specific grammar, a DSL within the DSL — that notation lives in a parser. The parser produces a pure data structure describing what was parsed. The parser does not know about morphisms, types, backends, or runtime.

This separation is what allows your custom notation to be tested in isolation, evolved without breaking the rest of the extension, or replaced entirely (e.g., a different surface syntax producing the same underlying spec) without rewriting the semantic layer.

### Rule 5: Domain semantics owns meaning

The conversion from a parsed spec to a `Morphism` is where your extension expresses what its domain *means*. This is the most important layer of the extension and the one most often gotten wrong.

The meaning of a domain construct must be expressed in terms of substrate combinators. If your extension wants to introduce a contraction over an index, that contraction means a fold using a domain-specific accumulator. The fold uses `cata` or `compose` or `par` — substrate operations. The accumulator is a `Morphism` you constructed. The contraction emerges from composing these substrate pieces.

The wrong pattern: "a contraction is a special thing that the runtime knows how to execute."

The right pattern: "a contraction is the composition of these specific substrate operations, and the substrate already knows how to execute compositions of those operations."

If you cannot express your domain construct in substrate terms, you have not finished thinking about what it means. The substrate is powerful enough to express any compositional structure. If you are stuck, the issue is almost always conceptual, not technical.

### Rule 6: Native execution belongs at the runtime boundary

If your extension needs to perform computation that the substrate's existing primitives do not provide — calling into a native library, invoking an external service, performing optimized numerical operations — that computation belongs in a registered backend primitive.

Backend primitives are declared in JSON backend specs (see `runtime/backends/`) and registered through the `BackendOps` machinery. Once registered, a backend primitive is available as a named morphism in your extension's semantic layer. The substrate handles dispatch, type checking, and integration with the rest of the system.

The wrong pattern: your extension's semantic layer calls `numpy.einsum` directly.

The right pattern: your extension's semantic layer constructs a `Morphism` whose realization invokes a registered `numpy.einsum` primitive, which is called by the runtime during execution.

This separation is what allows your extension to work across backends. If you write `import numpy` in your semantic layer, your extension only works for numpy. If you depend only on registered primitives by name, your extension works for any backend that registers those primitives.

### Rule 7: Custom optimization is fine; bypassing the substrate is not

Some domains have specialized algorithms that the substrate does not know about — query plan optimization, contraction order optimization, music theory voice-leading rules, legal precedent search heuristics. Extensions can and should implement these.

The optimization belongs in your extension's structure or semantic layer, operating on substrate objects. An optimizer takes a `Morphism` or a domain-specific spec and produces a better `Morphism`. It does not bypass realization. It does not produce its own kind of executable artifact. It produces a substrate object that the substrate then handles normally.

The wrong pattern: your extension has its own `execute()` function that runs optimized code directly.

The right pattern: your extension has an `optimize()` function that takes a `Morphism` and returns an equivalent but more efficient `Morphism`, which is then executed by the substrate's standard `run()`.

### Rule 8: Extensions are composable by default

Because every extension produces substrate objects, two extensions can be combined without coordination between their authors. A morphism produced by extension A can be composed with a morphism produced by extension B using ordinary `>>`. A functor defined by extension A can act on morphisms from extension B.

This composability is automatic if you follow rules 1 through 7. It is a *consequence* of obeying the contract, not a separate concern. The only way to break it is to violate the contract.

A specific implication: do not introduce extension-private types that other extensions cannot interoperate with. If your extension needs to represent something, represent it as a substrate object. If you need to attach metadata to a morphism, find a way that other extensions can ignore safely.

### Rule 9: Custom syntax integrates through the parser, not around it

If your extension provides surface syntax that should appear in unialg source files, integrate through the parser's extension points rather than building a parallel parser.

The recommended pattern is: your extension provides a function that takes a string (in your custom notation) and returns a substrate object. unialg source files use the existing `let` and `shape` declarations, and call into your function via a normal morphism reference or shape reference. The custom syntax appears inside a string literal or a special form recognized by your extension.

If your extension genuinely needs new top-level declaration keywords — beyond `let`, `shape`, and `load` — that requires a discussion about whether the substrate should support those keywords as a built-in extension point. Adding new declaration keywords ad hoc breaks the uniformity of the surface syntax. Most extensions do not need this; the few that do should be careful.

**Activation:** Extensions must be explicitly enabled before their syntax is available.
Use `extensions.enable("tensors")` in Python or `load extension tensors` in the DSL.
Each extension module exposes a `_register()` function that registers its keyword
handlers, expression handlers, and domain protocol. `is_enabled(name)` checks activation
status. Auto-registration on import is not supported.

### Rule 10: Completion requires layered tests

A complete extension has tests at each layer that has code. Specifically:

- If your extension parses, it has parser tests that verify parsed output for given input strings.
- If your extension constructs morphisms, it has semantic tests that verify the type signatures and structure of constructed morphisms.
- If your extension registers backend primitives, it has runtime tests that verify the primitives produce correct results for sample inputs.
- If your extension has optimizations, it has tests that verify the optimizations preserve semantics (the optimized morphism produces the same result as the unoptimized one).
- In addition, there must be at least one end-to-end test that exercises the full pipeline from surface input to executed output.

A notebook demo is not a test. A working example is not a test. Tests verify specific claims about behavior. An extension is not done until its claims are testable and tested.

### Rule 11: Documentation names the boundary

Every extension must include a brief document — a `README.md` or similar — that names which file handles which layer. The minimum content:

- Which file parses (if any)
- Which file constructs substrate objects
- Which files handle structure or runtime concerns
- Which files are notes or experiments rather than load-bearing code
- What this extension imports from the substrate and what it does not

This is enforcement by transparency. If an extension cannot describe its boundary clearly, that usually means the boundary is not clear, and the extension probably violates one of the earlier rules.

---

## What can a plugin actually do?

A plugin can:

- Add a new domain-specific notation, parsed into a domain-specific specification
- Provide functions that construct `Morphism`, `Functor`, `Optic`, or `Carrier` objects from domain inputs
- Register backend primitives required by the domain
- Implement domain-specific optimization passes over substrate objects
- Provide pre-built morphisms, functors, or optics that capture common patterns in the domain
- Expose its own examples, tutorials, and tests
- Be loaded alongside any other plugin without coordination

A plugin cannot:

- Modify the substrate (`semantics/`, `structure/`, `runtime/`, `main.py`, `objects.py`)
- Bypass the substrate to call native code directly
- Introduce a parallel evaluator or execution path
- Require modifications to the parser's top-level grammar (unless specifically discussed and accepted as a substrate extension)
- Hide its execution behavior — every effect must flow through the standard `run()` pipeline

If a plugin needs to do one of the things in the "cannot" list, the right path is to propose a substrate change as a separate discussion, complete the substrate change with full review, and then build the plugin against the updated substrate.

---

## Layer responsibilities for extensions

This section translates the substrate's layer model into guidance for extension authors.

### Surface notation (if your extension provides syntax)

Lives in: a parser file in your extension's directory.

Responsibilities:
- Tokenize and parse strings written in your domain's notation
- Produce a pure data structure describing what was parsed
- Validate syntactic well-formedness
- Report parse errors with useful location information

Does not:
- Import from `semantics/`, `structure/`, or `runtime/`
- Construct `Morphism` objects
- Know about types or backends
- Perform any computation beyond parsing

### Domain semantics (almost every extension has this)

Lives in: one or more files that take parsed specs (or domain inputs) and produce substrate objects.

Responsibilities:
- Convert parsed specs into `Morphism`, `Functor`, `Optic`, or `Carrier` objects
- Express domain meaning in terms of substrate combinators
- Provide convenience constructors for common patterns in the domain
- Validate that constructed objects are well-formed at the type level

Does not:
- Import from `runtime/`
- Call native backends directly
- Construct Hydra terms (that is `structure/realize.py`'s job)
- Know about specific backend implementations

### Domain structure (only if needed)

Lives in: a file for custom realization or lowering, if your extension needs Hydra term construction beyond what existing realization handles.

Responsibilities:
- Provide custom recursion schemes or term shapes specific to the domain
- Wire domain-specific primitives into the realization pipeline

Does not exist if your extension can express itself entirely through existing semantic combinators. Most extensions do not need this layer. If you think you do, examine the substrate's existing combinators carefully first — the substrate is more expressive than it looks.

### Runtime adapters (only if needed)

Lives in: a file that registers backend primitives required by your extension.

Responsibilities:
- Declare which native operations your extension requires
- Provide codec specifications for any domain-specific value encodings
- Register backend primitives with the standard `BackendOps` machinery

Does not exist if your extension uses only operations already provided by existing backends. The runtime layer is the boundary to the outside world; an extension only adds to it if the outside world has something the substrate does not already know how to invoke.

---

## A worked outline

This section provides a generic template for how an extension is organized. Read it as scaffolding, not as a specific recipe. Adapt the file structure to your domain's needs.

```
src/unialg/<domain>/
  __init__.py          # public API of the extension
  notation.py          # parser (if domain has custom notation)
  semantics.py         # spec → Morphism / Functor / Optic construction
  primitives.py        # backend primitive declarations (if needed)
  optimize.py          # domain-specific optimization passes (if any)
  README.md            # boundary documentation
  examples/            # example programs using the extension
  tests/               # layered tests
```

Some extensions are smaller and fit in a single file. Some need additional layers. The structure above is illustrative.

The `__init__.py` exposes a small public API: the functions and types that users of the extension call. Everything else is internal.

A typical user interaction with an extension:

```python
from unialg import compile_program
from unialg.<domain> import some_extension_function

# Either: use the extension's helpers to construct morphisms directly
m = some_extension_function(domain_input)
result = compile_program(...).run(m, arg)

# Or: reference the extension's helpers from a unialg source file via let
program = compile_program("""
load <backend>
let f = <extension-provided morphism reference>
""")
```

Both patterns are valid. Which one your extension supports depends on whether it provides constructive helpers, surface syntax, or both.

---

## Common pitfalls

These are mistakes that extension authors make. Each has a fix that aligns the extension with the contract.

### Direct native calls in the semantic layer

If `import numpy` (or `import torch`, etc.) appears in your semantic layer, your extension has crossed a layer boundary. The fix is to identify what native operation you needed and register it as a backend primitive. Then call it through a substrate morphism instead.

### Bypassing realization

If your extension has its own `evaluate()` or `execute()` function that runs domain logic without going through `realize` and `run`, you have a parallel evaluator. The fix is to express the same logic as a substrate composition and let the standard pipeline execute it.

### Adding to substrate files

If your extension requires you to add a case to `semantics/morphisms.py` or `structure/realize.py`, you are not extending — you are modifying. The fix is to express what you need as a composition of existing primitives. If genuinely impossible, propose a substrate change as a separate concern.

### Hidden state

If your extension maintains module-level state that other extensions or future runs depend on, you have introduced coupling. The fix is to thread state through arguments or use the substrate's runtime store mechanism, which has well-defined lifetime semantics.

### Conflating spec and semantics

If your parsed specification format duplicates information that the semantic layer also tracks (types, arities, monads), you have two sources of truth. The fix is to keep the spec minimal — just what the parser produces — and let the semantic layer derive everything else when constructing the morphism.

### Optimization that breaks composability

If your optimizer produces objects that look like morphisms but cannot be composed with morphisms from other extensions, the optimizer is producing the wrong output type. The fix is to ensure the optimizer's output is an ordinary `Morphism` indistinguishable in interface from any other morphism.

### Surface syntax that requires substrate changes

If you find yourself wanting to add new keywords to the top-level parser, your extension is asking for too much. The fix in most cases is to use existing `let` declarations and embed your custom syntax inside string arguments or special forms recognized by your extension's helper functions.

---

## When the contract is wrong

The contract above is opinionated. It will occasionally be wrong for a specific domain.

When that happens, do not silently violate the contract. Instead:

1. Identify exactly which rule fails and why.
2. Describe what would need to change in the substrate to accommodate your domain cleanly.
3. Propose that substrate change as a separate discussion, with full review of how it affects other extensions and the system as a whole.
4. If the substrate change is accepted, complete it before building your extension.
5. If the substrate change is rejected, find a different way to express your domain within the existing contract.

The contract is not sacred. It is a working agreement that reflects current understanding. If your extension reveals that the understanding is incomplete, that is valuable information. But the way to act on it is to update the contract for everyone, not to write an extension that quietly violates it.

The reason this matters: every extension that violates the contract makes future extensions harder. Composability degrades. Surprises accumulate. Eventually the system becomes one where extensions cannot be combined and each extension needs to know about every other extension. That is the failure mode the contract prevents.

---

## Examples of valid extensions

To make the contract concrete, here are types of extensions that fit within it cleanly. None of these are tensor algebra. The point is that the contract supports a wide range of domains.

A **music composition extension** might parse score notation into a spec, construct morphisms over a category of musical phrases with composition meaning sequence and pairing meaning counterpoint, register backend primitives for audio synthesis, and provide optimization for voice-leading constraints.

A **legal reasoning extension** might parse case citations into a spec, construct morphisms over a category of legal claims with composition meaning argument chaining and pairing meaning conjunctive claims, register backend primitives for case retrieval and citation analysis, and provide optimization for argument structure.

A **chemical synthesis extension** might parse reaction equations into a spec, construct morphisms over a category of molecular structures with composition meaning reaction sequences, register backend primitives for chemical property lookup and feasibility checking, and provide optimization for synthesis route planning.

A **cognitive architecture extension** might parse production rules into a spec, construct morphisms over a category of mental states with composition meaning rule chaining, register backend primitives for memory lookup, and provide optimization for activation spreading.

An **ecological modeling extension** might parse species interaction matrices into a spec, construct morphisms over a category of populations with composition meaning temporal evolution, register backend primitives for environmental data, and provide optimization for stability analysis.

In each case, the structure of the extension is the same: parse, construct, possibly register backend primitives, possibly optimize, always elaborating into the substrate. The domain differs entirely; the architecture does not.

This is the test of whether the contract is the right one. A contract that worked for tensors but not for music would be a contract that smuggled in tensor assumptions. The contract above intentionally does not. If your domain fits the pattern *parse → construct → elaborate*, the substrate is for you. If it does not, the substrate may genuinely not fit — and that is information worth surfacing.
