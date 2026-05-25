"""Tests for native tensor backend execution via RuntimeStore."""
import pytest
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import hydra.dsl.meta.phantoms as P

from unialg.main import compile_morphism
from unialg.syntax.parse import parse_program
from unialg.semantics.construct import construct_program
from unialg.semantics.morphisms import Morphism
from unialg.syntax import expressions as expr
from unialg.runtime import BackendOps, RuntimeStore, coder_for_type, type_from_spec
from unialg.runtime.backend import load_spec, register_backend_primitive


@pytest.fixture
def numpy_backend():
    return BackendOps.from_spec("src/unialg/runtime/backends/numpy.json")


@pytest.fixture
def env(numpy_backend):
    e = {}
    for name, bp in numpy_backend.primitives.items():
        e[name] = Morphism(
            node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type),
            aux_primitives=(bp.primitive,),
        )
    return e


@pytest.fixture
def store(numpy_backend):
    return numpy_backend.store


def _compile_let(src, env):
    parsed = parse_program(src)
    constructed = construct_program(parsed, env)
    morphism_name = list(constructed.morphisms.keys())[-1]
    return compile_morphism(constructed.morphisms[morphism_name])


def _run_unary(prog, store, arr):
    store.reset()
    key = store.put(arr)
    result = prog.run(P.binary(key).value)
    return store.get(result)


def _run_binary(prog, store, a, b):
    store.reset()
    key_a = store.put(a)
    key_b = store.put(b)
    result = prog.run(P.pair(P.binary(key_a), P.binary(key_b)).value)
    return store.get(result)


class TestUnaryOps:
    def test_tanh(self, env, store):
        prog = _compile_let("let f = tanh", env)
        arr = np.array([1.0, 2.0, 3.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.tanh(arr))

    def test_exp(self, env, store):
        prog = _compile_let("let f = exp", env)
        arr = np.array([0.0, 1.0, -1.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.exp(arr))

    def test_log(self, env, store):
        prog = _compile_let("let f = log", env)
        arr = np.array([1.0, 2.718, 10.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.log(arr))

    def test_sqrt(self, env, store):
        prog = _compile_let("let f = sqrt", env)
        arr = np.array([4.0, 9.0, 16.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.sqrt(arr))

    @given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=1, max_size=10))
    @settings(max_examples=10)
    def test_tanh_property(self, values):
        ops = BackendOps.from_spec("src/unialg/runtime/backends/numpy.json")
        store = ops.store
        e = {name: Morphism(node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type), aux_primitives=(bp.primitive,)) for name, bp in ops.primitives.items()}
        prog = _compile_let("let f = tanh", e)
        arr = np.array(values)
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.tanh(arr))


class TestBinaryOps:
    def test_add(self, env, store):
        prog = _compile_let("let f = add", env)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, a + b)

    def test_multiply(self, env, store):
        prog = _compile_let("let f = multiply", env)
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, a * b)

    def test_subtract(self, env, store):
        prog = _compile_let("let f = subtract", env)
        a = np.array([10.0, 20.0])
        b = np.array([3.0, 7.0])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, a - b)


class TestComposition:
    def test_add_then_tanh(self, env, store):
        prog = _compile_let("let f = add >> tanh", env)
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, np.tanh(a + b))

    def test_chain_unary(self, env, store):
        prog = _compile_let("let f = exp >> log", env)
        arr = np.array([1.0, 2.0, 3.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.log(np.exp(arr)))

    def test_multiply_then_sqrt(self, env, store):
        prog = _compile_let("let f = multiply >> sqrt", env)
        a = np.array([4.0, 9.0])
        b = np.array([4.0, 9.0])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, np.sqrt(a * b))


class TestStoreLifecycle:
    def test_reset_clears(self, env, store):
        arr = np.array([1.0])
        key = store.put(arr)
        store.reset()
        with pytest.raises(KeyError):
            store.get(key)

    def test_multiple_runs_independent(self, env, store):
        prog = _compile_let("let f = tanh", env)
        for i in range(5):
            arr = np.array([float(i)])
            out = _run_unary(prog, store, arr)
            assert np.allclose(out, np.tanh(arr))


class TestMixedStoreBoundary:
    def test_binary_arg_scalar_result_uses_store_only_for_input(self):
        store = RuntimeStore()
        arg_type = type_from_spec("BINARY")
        result_type = type_from_spec("FLOAT")
        bp = register_backend_primitive(
            "unialg.backend.sum_scalar",
            lambda arr: float(np.sum(arr)),
            arg_type,
            1,
            arg_coder=coder_for_type(arg_type),
            result_coder=coder_for_type(result_type),
            result_type=result_type,
            store=store,
        )
        ops = BackendOps({"sum_scalar": bp}, binary_adapter=object(), store=store)
        morphism = Morphism(
            node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type),
            aux_primitives=(bp.primitive,),
        )
        prog = compile_morphism(morphism, backend=ops)

        result = prog.run(np.array([1.0, 2.0, 3.0]))

        assert result == 6.0

    def test_load_spec_resolves_operation_paths_immediately(self):
        spec = {
            "backend": "broken",
            "operations": {
                "missing": {
                    "kind": "test",
                    "path": "math.not_a_real_function",
                    "arity": 1,
                    "arg_type": "FLOAT",
                    "result_type": "FLOAT",
                }
            },
        }
        with pytest.raises(ValueError, match="broken.*missing.*not_a_real_function"):
            load_spec(spec)

    def test_binary_specs_get_store_without_binary_adapter(self):
        spec = {
            "backend": "handles",
            "operations": {
                "sqrt": {
                    "kind": "test",
                    "path": "math.sqrt",
                    "arity": 1,
                    "arg_type": "BINARY",
                    "result_type": "BINARY",
                }
            },
        }
        _, adapter, store = load_spec(spec)

        assert adapter is None
        assert isinstance(store, RuntimeStore)


class TestTypedLiteralArguments:
    @pytest.mark.parametrize(
        ("name", "type_spec", "source", "fn", "expected"),
        [
            ("with_float", "FLOAT", "0.5", lambda arr, v: arr + v, np.array([1.5, 2.5])),
            ("with_bool", "BOOL", "true", lambda arr, v: arr if v else -arr, np.array([1.0, 2.0])),
            ("with_string", "STRING", "true", lambda arr, v: arr if v == "true" else -arr, np.array([1.0, 2.0])),
        ],
    )
    def test_scalar_literal_type_is_selected_by_declared_argument(
        self, name, type_spec, source, fn, expected,
    ):
        store = RuntimeStore()
        binary = type_from_spec("BINARY")
        config = type_from_spec(type_spec)
        bp = register_backend_primitive(
            f"unialg.backend.{name}",
            fn,
            None,
            2,
            arg_coder=None,
            arg_types=(binary, config),
            arg_coders=(coder_for_type(binary), coder_for_type(config)),
            result_coder=coder_for_type(binary),
            result_type=binary,
            store=store,
        )
        ops = BackendOps({name: bp}, store=store)
        env = {
            name: Morphism(
                node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type),
                aux_primitives=(bp.primitive,),
            )
        }
        morphism = construct_program(
            parse_program(f"let f = {name}(x, '{source}')"), env,
        ).morphisms["f"]
        prog = compile_morphism(morphism, backend=ops)

        assert np.allclose(prog.run(np.array([1.0, 2.0])), expected)

    def test_multiple_configured_arguments_follow_declared_arity(self):
        store = RuntimeStore()
        binary = type_from_spec("BINARY")
        integer = type_from_spec("INT")
        boolean = type_from_spec("BOOL")
        bp = register_backend_primitive(
            "unialg.backend.shift_if",
            lambda arr, amount, enabled: arr + amount if enabled else arr,
            None,
            3,
            arg_coder=None,
            arg_types=(binary, integer, boolean),
            arg_coders=(
                coder_for_type(binary),
                coder_for_type(integer),
                coder_for_type(boolean),
            ),
            result_coder=coder_for_type(binary),
            result_type=binary,
            store=store,
        )
        ops = BackendOps({"shift_if": bp}, store=store)
        env = {
            "shift_if": Morphism(
                node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type),
                aux_primitives=(bp.primitive,),
            )
        }
        morphism = construct_program(
            parse_program("let f = shift_if(x, '2', 'true')"), env,
        ).morphisms["f"]
        prog = compile_morphism(morphism, backend=ops)

        assert np.allclose(prog.run(np.array([1.0, 2.0])), np.array([3.0, 4.0]))
