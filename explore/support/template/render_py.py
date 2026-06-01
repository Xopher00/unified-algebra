#!/usr/bin/env python3
"""
Render arch.py and __init__.py from an ArchSpec JSON file.

Usage:
    python explore/support/template/render_py.py <spec.json>
"""

import json
import os
import sys


# ── helpers ───────────────────────────────────────────────────────────────────

def to_pascal(label: str) -> str:
    return "".join(w.capitalize() for w in label.split("_"))


def to_camel(label: str) -> str:
    parts = label.split("_")
    return parts[0] + "".join(w.capitalize() for w in parts[1:])


def render_expr(expr: dict, state_var: str, input_var: str) -> str:
    """Translate one ANF binding expression to Python."""
    tag = expr["tag"]

    def sub(arg: str) -> str:
        if arg == state_var:
            return "_h"
        if arg == input_var:
            return "inp"
        return arg

    if tag == "Contraction":
        eq   = expr["equation"]
        args = [sub(a) for a in expr["args"]]
        return f'lib.einsum("{eq}", {", ".join(args)})'
    elif tag == "ElemOp":
        op   = expr["op"]
        args = [sub(a) for a in expr["args"]]
        if op == "add":
            return " + ".join(args)
        elif op == "multiply":
            return " * ".join(args)
        else:
            return f"{op}({', '.join(args)})"
    elif tag == "Activation":
        kind = expr["kind"]
        arg  = sub(expr["arg"])
        return f"lib.{kind}({arg})"
    else:
        raise ValueError(f"Unknown CellExpr tag: {tag!r}")


# ── numpy reference generation ────────────────────────────────────────────────

def numpy_ref_ana(cell: dict) -> str:
    """Generate numpy reference for AnaArch (coalgebra unroll)."""
    res    = cell["result"]
    sv     = res["state_var"]
    iv     = res["input_var"]
    params = cell["params"]
    bmap   = {b["name"]: b["expr"] for b in cell["bindings"]}

    out_bs  = res["output_bindings"]
    out_v   = res["output"]
    ns_bs   = res["next_state_bindings"]
    ns_v    = res["next_state"]

    def py(name: str) -> str:
        return render_expr(bmap[name], sv, iv)

    lines = [
        f"def _numpy_reference(backend, {', '.join(params)}, s0, inputs):",
        "    lib = backend.framework",
        "    outputs = []",
        "    _h = s0",
        "    for inp in inputs:",
    ]
    for n in out_bs:
        lines.append(f"        {n} = {py(n)}")
    lines.append(f"        outputs.append(np.array({out_v}))")
    for n in ns_bs:
        lines.append(f"        {n} = {py(n)}")
    lines.append(f"        _h = {ns_v}")
    lines.append("    return outputs")
    return "\n".join(lines)


def numpy_ref_cata_const(cell: dict) -> str:
    """Generate numpy reference for CataConst (list-fold)."""
    res    = cell["result"]
    params = cell["params"]
    bmap   = {b["name"]: b["expr"] for b in cell["bindings"]}

    base      = res["base"]
    step_vars = res["step_vars"]
    step_bs   = res["step_bindings"]
    step_res  = res["step_result"]
    # step_vars[0] = element var (Haskell 'a'), step_vars[1] = accumulator (Haskell 's')
    acc_var  = step_vars[1] if len(step_vars) > 1 else "_s"
    elem_var = step_vars[0]

    def py(name: str) -> str:
        return render_expr(bmap[name], acc_var, elem_var)

    lines = [
        f"def _numpy_reference(backend, {', '.join(params)}, inputs):",
        "    lib = backend.framework",
        f"    {acc_var} = {base}",
        f"    for {elem_var} in inputs:",
    ]
    for n in step_bs:
        lines.append(f"        {n} = {py(n)}")
    lines.append(f"        {acc_var} = {step_res}")
    lines.append("    return _s")
    return "\n".join(lines)


def numpy_ref_cata_fn(cell: dict) -> str:
    """Generate numpy reference for CataFn (tree-fold, no input sequence)."""
    res    = cell["result"]
    params = cell["params"]
    bmap   = {b["name"]: b["expr"] for b in cell["bindings"]}

    bv       = res["base_var"]
    base_bs  = res["base_bindings"]
    base     = res["base"]
    step_vs  = res["step_vars"]
    step_bs  = res["step_bindings"]
    step_res = res["step_result"]

    def py_base(name: str) -> str:
        return render_expr(bmap[name], bv, "__unused__")

    def py_step(name: str) -> str:
        sv, iv = step_vs[0], step_vs[1] if len(step_vs) > 1 else "__unused__"
        return render_expr(bmap[name], sv, iv)

    base_fn_lines = [f"    def _base({bv}):"]
    for n in base_bs:
        base_fn_lines.append(f"        {n} = {py_base(n)}")
    base_fn_lines.append(f"        return {base}")

    step_fn_lines = [f"    def _step({', '.join(step_vs)}):"]
    for n in step_bs:
        step_fn_lines.append(f"        {n} = {py_step(n)}")
    step_fn_lines.append(f"        return {step_res}")

    lines = [
        f"def _numpy_reference(backend, {', '.join(params)}, tree):",
        "    lib = backend.framework",
    ]
    lines += base_fn_lines + step_fn_lines
    lines.append("    return _fold(tree, _base, _step)")
    return "\n".join(lines)


def numpy_ref_pure(cell: dict) -> str:
    """Generate numpy reference for ResPure (no recursion)."""
    res    = cell["result"]
    params = cell["params"]
    bmap   = {b["name"]: b["expr"] for b in cell["bindings"]}

    iv     = res["input_var"]
    pure_bs = res["pure_bindings"]
    result  = res["result"]

    def py(name: str) -> str:
        return render_expr(bmap[name], iv, "__unused__")

    lines = [
        f"def _numpy_reference(backend, {', '.join(params)}, x):",
        "    lib = backend.framework",
    ]
    for n in pure_bs:
        lines.append(f"    {n} = {py(n)}")
    lines.append(f"    return np.array({result})")
    return "\n".join(lines)


def numpy_ref(cell: dict) -> str:
    tag = cell["result"]["tag"]
    if tag == "Ana":
        return numpy_ref_ana(cell)
    elif tag == "CataConst":
        return numpy_ref_cata_const(cell)
    elif tag == "CataFn":
        return numpy_ref_cata_fn(cell)
    elif tag == "Pure":
        return numpy_ref_pure(cell)
    else:
        raise ValueError(f"Unknown result tag: {tag!r}")


# ── framework reference stubs ─────────────────────────────────────────────────

def tf_stub(ref: dict | None, cell: dict) -> str:
    if ref is None:
        return _manual_stub("_tf_reference", "TFBackend", cell)
    tag = ref["tag"]
    params = cell["params"]
    if tag == "SingleClass":
        cls    = ref["class"]
        return "\n".join([
            f"def _tf_reference(backend, {', '.join(params)}, s0, inputs):",
            "    tf = backend.framework",
            "    hidden = int(s0.shape[0])",
            "    inp_size = int(inputs[0].shape[0])",
            f"    cell = {cls}",
            "    cell.build((None, inp_size))",
            "    # TODO: assign weights according to TF row-vector convention",
            "    h = s0[None]",
            "    outputs = []",
            "    for inp in inputs:",
            "        outputs.append(np.array(h.numpy().squeeze()))",
            "        h, _ = cell(inp[None], [h])",
            "    return outputs",
        ])
    return _manual_stub("_tf_reference", "TFBackend", cell)


def torch_stub(ref: dict | None, cell: dict) -> str:
    if ref is None:
        return _manual_stub("_torch_reference", "TorchBackend", cell)
    tag = ref["tag"]
    params = cell["params"]
    if tag == "SingleClass":
        cls    = ref["class"]
        return "\n".join([
            f"def _torch_reference(backend, {', '.join(params)}, s0, inputs):",
            "    torch = backend.framework",
            "    hidden = s0.shape[0]",
            "    inp_size = inputs[0].shape[0]",
            f"    cell = {cls}.double()",
            "    with torch.no_grad():",
            "        pass  # TODO: assign weights according to Torch column-vector convention",
            "    h = s0.unsqueeze(0)",
            "    outputs = []",
            "    for inp in inputs:",
            "        outputs.append(np.array(h.squeeze().detach()))",
            "        h = cell(inp.unsqueeze(0), h)",
            "    return outputs",
        ])
    return _manual_stub("_torch_reference", "TorchBackend", cell)


def _manual_stub(fn_name: str, backend_name: str, cell: dict) -> str:
    params = cell["params"]
    return "\n".join([
        f"def {fn_name}(backend, {', '.join(params)}, s0, inputs):",
        f"    # TODO: implement {backend_name} reference",
        "    raise NotImplementedError",
    ])


# ── test / hypothesis generation ─────────────────────────────────────────────

def strategy_stub(label: str, cell: dict) -> str:
    sname  = to_camel(label) + "_inputs"
    params_comment = ", ".join(cell["params"])
    return "\n".join([
        "@st.composite",
        f"def {sname}(draw, backend):",
        f"    # TODO: draw dimension variables; construct tensors for: {params_comment}",
        "    # Example for a hidden-dim-h architecture:",
        "    # h = draw(st.sampled_from(HIDDEN_DIMS))",
        "    # w = backend.random_matrix(draw, h, h)",
        "    # s0 = backend.random_vector(draw, h)",
        "    # inputs = [backend.random_vector(draw, h) for _ in range(n)]",
        "    raise NotImplementedError",
    ])


def test_class(label: str, cell: dict) -> str:
    cls_name  = "Test" + to_pascal(label)
    seed_name = to_camel(label)
    sname     = seed_name + "_inputs"
    params    = cell["params"]
    result    = cell["result"]
    tag       = result["tag"]

    if tag == "Ana":
        params_load = ", ".join(params + ["s0", "inputs", "_n"])
        return "\n".join([
            f"class {cls_name}:",
            "",
            "    @given(data=st.data())",
            "    @settings(max_examples=50, **HYPO)",
            f"    def test_{label}(self, spec, data):",
            "        backend = spec.backend",
            f"        {params_load} = data.draw({sname}(backend))",
            "",
            "        fn = spec.load(GENERATED_ROOT)",
            f"        gen = _take_n(fn, {', '.join(params)}, s0, inputs)",
            f"        ref = spec.reference(backend, {', '.join(params)}, s0, inputs)",
            "",
            "        assert len(gen) == _n",
            "        for k, (g, r) in enumerate(zip(gen, ref)):",
            "            assert backend.allclose(g, r, atol=1e-5), \\",
            f'                f"[{{backend.name}}] mismatch at step {{k + 1}} of {{_n}}"',
        ])
    else:
        params_load = ", ".join(params + ["x"])
        return "\n".join([
            f"class {cls_name}:",
            "",
            "    @given(data=st.data())",
            "    @settings(max_examples=50, **HYPO)",
            f"    def test_{label}(self, spec, data):",
            "        backend = spec.backend",
            f"        {params_load} = data.draw({sname}(backend))",
            "",
            "        fn = spec.load(GENERATED_ROOT)",
            "        gen = fn(x)",
            f"        ref = spec.reference(backend, {', '.join(params)}, x)",
            "",
            "        assert backend.allclose(gen, ref, atol=1e-5), \\",
            f'            f"[{{backend.name}}] mismatch"',
        ])


# ── take_n helper ─────────────────────────────────────────────────────────────

def take_n_helper(cell: dict) -> str:
    params = cell["params"]
    result = cell["result"]
    if result["tag"] != "Ana":
        return ""
    return "\n".join([
        f"def _take_n(fn, {', '.join(params)}, s0, inputs):",
        "    \"\"\"Unroll the coalgebra for len(inputs) steps.\"\"\"",
        "    outputs = []",
        f"    step = fn({', '.join(params)}, s0)",
        "    for inp in inputs:",
        "        output, cont = step",
        "        outputs.append(np.array(output))",
        "        step = cont(inp)",
        "    return outputs",
    ])


# ── full arch.py ──────────────────────────────────────────────────────────────

def render_arch_py(spec: dict) -> str:
    label = spec["label"]
    cell  = spec["cell"]
    ref   = spec["ref"]
    arch  = spec["arch"]

    poly_f     = arch["poly_f"]
    arch_class = arch["class"]

    seed_fn    = label                   # matches what the Haskell module calls "seed.<label>"
    seed_step  = label + "_step"

    tf_ref_entry    = ref.get("tensorflow")
    torch_ref_entry = ref.get("torch")

    parts = []

    # docstring
    parts.append(f'"""\n{to_pascal(label)} — {arch_class}.\n\nFunctor: {poly_f}\n"""')
    parts.append("")

    # imports
    parts.append("import numpy as np")
    parts.append("import pytest")
    parts.append("from hypothesis import given, settings, strategies as st")
    parts.append("")
    parts.append("from backends import (")
    parts.append("    BackendSpec,")
    parts.append("    NumpyBackend,")
    parts.append("    TFBackend,")
    parts.append("    TorchBackend,")
    parts.append("    HYPO,")
    parts.append("    arch_generated_root,")
    parts.append(")")
    parts.append("")
    parts.append("GENERATED_ROOT = arch_generated_root(__file__)")
    parts.append("")

    # dimension constants (placeholder — agent fills with real values)
    parts.append("HIDDEN_DIMS = [2, 3]")
    parts.append("OUTPUT_DIMS = [1, 2]")
    parts.append("INPUT_DIMS  = [2, 3]")
    parts.append("MAX_STEPS   = 4")
    parts.append("")

    # semiring identity constants — derived from the spec, not hardcoded
    semirings_seen: dict[str, dict] = {}
    for b in cell["bindings"]:
        expr = b["expr"]
        if expr["tag"] == "Contraction":
            sr = expr["semiring"]
            semirings_seen[sr["label"]] = sr
    for sr in semirings_seen.values():
        parts.append(f"SR_{sr['label'].upper()}_ZERO = {sr['zero']}")
        parts.append(f"SR_{sr['label'].upper()}_ONE  = {sr['one']}")
    if semirings_seen:
        parts.append("")

    # take_n helper (Ana only)
    helper = take_n_helper(cell)
    if helper:
        parts.append(helper)
        parts.append("")

    # numpy reference
    parts.append(numpy_ref(cell))
    parts.append("")

    # tf reference
    parts.append(tf_stub(tf_ref_entry, cell))
    parts.append("")

    # torch reference
    parts.append(torch_stub(torch_ref_entry, cell))
    parts.append("")

    # BACKENDS
    parts.append("BACKENDS = [")
    for backend_name, ref_fn in [("numpy", "_numpy_reference"),
                                  ("tensorflow", "_tf_reference"),
                                  ("torch", "_torch_reference")]:
        parts.append(f"    BackendSpec(")
        parts.append(f"        {backend_name.capitalize() if backend_name != 'tensorflow' else 'TF'}Backend(),")
        parts.append(f'        module="seed.{seed_fn}",')
        parts.append(f'        fn="{seed_step}",')
        parts.append(f"        reference={ref_fn},")
        parts.append(f"    ),")
    parts.append("]")
    parts.append("")

    # Hypothesis strategy stub
    parts.append(strategy_stub(label, cell))
    parts.append("")

    # pytest fixture
    parts.append("@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)")
    parts.append("def spec(request):")
    parts.append("    return request.param")
    parts.append("")

    # test class
    parts.append(test_class(label, cell))
    parts.append("")

    return "\n".join(parts)


# ── __init__.py ───────────────────────────────────────────────────────────────

def render_init_py() -> str:
    return ""  # empty; pytest discovers arch.py directly


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: render_py.py <spec.json>")
    spec_path = sys.argv[1]
    with open(spec_path) as f:
        spec = json.load(f)

    label    = spec["label"]
    arch_dir = os.path.join("explore", "archs", label)
    os.makedirs(arch_dir, exist_ok=True)

    arch_py = os.path.join(arch_dir, "arch.py")
    with open(arch_py, "w") as f:
        f.write(render_arch_py(spec))
    print(f"Wrote {arch_py}")

    init_py = os.path.join(arch_dir, "__init__.py")
    with open(init_py, "w") as f:
        f.write(render_init_py())
    print(f"Wrote {init_py}")


if __name__ == "__main__":
    main()
