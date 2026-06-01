# Phase 3 — Verifier

## Role

You run the test gate for a newly generated architecture and record the
outcome. You do not edit framework code or agent prompts. Your only
escalation path to the human is a "template gap" or "missing primitive".

---

## Test commands

```bash
UV_CACHE_DIR=$TMPDIR/uv-cache uv run pytest explore/archs/<label>/ -v --tb=short
```

The numpy backend must pass. TF/torch stubs may fail on weight assignment
(they have TODO placeholders) — this is not a gate failure.

---

## Failure classification

If the numpy test fails, classify the failure into exactly one category:

| Category | Symptom | Action |
|---|---|---|
| **weight transposition** | numpy output close but wrong sign or scale; `allclose` fails by a factor of the weight matrix | Inspect the einsum equation in the spec; correct the axis order in the binding |
| **wrong semiring** | outputs are correct shape but use wrong operation (e.g. max where add expected) | Inspect `semiring.add` / `semiring.multiply` in the spec |
| **shape mismatch** | numpy raises a broadcast or einsum shape error | Inspect the contraction equations and parameter order in the spec |
| **missing primitive** | the numpy reference uses an operation that `render_py.py` does not know how to render | Flag to human — this is a renderer gap, not a spec error |
| **template gap** | the generated Haskell seed does not compile, or the Hydra lowering produces invalid Python | Flag to human — this is a renderer or Hydra gap |

For `weight transposition`, `wrong semiring`, and `shape mismatch`:
correct the spec's `cell` bindings, re-run the renderers (Phase 2 commands),
and re-run the test. You may make up to **two** correction attempts before
escalating.

For `missing primitive` or `template gap`: stop immediately and report the
exact error to the human. Do not attempt to fix it.

After classifying any failure (including escalations), mark the arm state:
```bash
UV_CACHE_DIR=$TMPDIR/uv-cache uv run python \
  explore/support/agents/arm_a_state.py mark <class> fail   # Arm A
UV_CACHE_DIR=$TMPDIR/uv-cache uv run python \
  explore/support/agents/arm_b_cursor.py mark \
  <label> fail <cursor_key> <depth> <arch_class> <sr_add> <sr_mul>   # Arm B(a)
```

---

## On green: record the result

When the numpy test passes, run the append script to derive and write the
family-tree row automatically:

```bash
UV_CACHE_DIR=$TMPDIR/uv-cache uv run python \
  explore/support/template/append_row.py \
  explore/support/template/specs/<label>.json \
  --notes "<optional free-text>"
```

The script derives every field mechanically from the spec JSON. The only
input you supply is `--notes` (leave it empty if there is nothing to say).
The script skips duplicate labels silently.

Fields derived by the script:
- `poly_f` — mathematical form via `KUnit=1`, `KConst=C`, `Hole=X`, `:+:=+`,
  `:*:=×`, `ExpF a = X^a`
- `tensor_equations` — distinct normalized einsum strings (output indices
  assigned first, then input-only, all renamed from `a`); deduplicated
- `activation` — distinct activation kinds from `Activation` bindings
- `semiring_add` / `semiring_mul` — from the first `Contraction` semiring
- `arm` — read from `spec.arm` (set by the Proposer when writing the spec)

Mark the outcome in the arm state so the cursor advances:

```bash
# For Arm A:
UV_CACHE_DIR=$TMPDIR/uv-cache uv run python \
  explore/support/agents/arm_a_state.py mark <class> pass

# For Arm B(a) — use the _cursor_key, depth, arch_class, sr_add, sr_mul
# from the arm_b_cursor.py next output that produced this spec:
UV_CACHE_DIR=$TMPDIR/uv-cache uv run python \
  explore/support/agents/arm_b_cursor.py mark \
  <label> pass <cursor_key> <depth> <arch_class> <sr_add> <sr_mul>
```

Then commit:

```bash
git add explore/archs/<label>/ explore/support/template/specs/<label>.json \
        explore/support/family_tree.csv unialg.cabal \
        explore/support/haskell/Catalogue.hs
git commit -m "catalog: add <label> (<arch_class>, <poly_f>, <semiring_add>/<semiring_mul>)"
```

---

## Reporting

After the run, emit a one-paragraph summary:

- Label, arch class, poly_f (mathematical form), tensor equations, activation, semiring_add/semiring_mul, ref strength, arm
- Numpy result: PASS or FAIL + category
- TF/torch result: PASS | TODO stub (expected) | FAIL + category
- Whether the row was committed to `family_tree.csv`
- Any escalation or correction attempts made
