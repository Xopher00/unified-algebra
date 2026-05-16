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
from unialg.runtime.backend import BackendOps


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


def _compile_route(src, env):
    parsed = parse_program(src)
    constructed = construct_program(parsed, env)
    route_name = list(constructed.routes.keys())[-1]
    return compile_morphism(constructed.routes[route_name])


def _run_unary(prog, store, arr):
    store.reset()
    key = store.put(arr)
    result = prog.run(P.binary(key).value)
    return store.get(result.value.value)


def _run_binary(prog, store, a, b):
    store.reset()
    key_a = store.put(a)
    key_b = store.put(b)
    result = prog.run(P.pair(P.binary(key_a), P.binary(key_b)).value)
    return store.get(result.value.value)


class TestUnaryOps:
    def test_tanh(self, env, store):
        prog = _compile_route("route f = tanh", env)
        arr = np.array([1.0, 2.0, 3.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.tanh(arr))

    def test_exp(self, env, store):
        prog = _compile_route("route f = exp", env)
        arr = np.array([0.0, 1.0, -1.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.exp(arr))

    def test_log(self, env, store):
        prog = _compile_route("route f = log", env)
        arr = np.array([1.0, 2.718, 10.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.log(arr))

    def test_sqrt(self, env, store):
        prog = _compile_route("route f = sqrt", env)
        arr = np.array([4.0, 9.0, 16.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.sqrt(arr))

    @given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=1, max_size=10))
    @settings(max_examples=10)
    def test_tanh_property(self, values):
        ops = BackendOps.from_spec("src/unialg/runtime/backends/numpy.json")
        store = ops.store
        e = {name: Morphism(node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type), aux_primitives=(bp.primitive,)) for name, bp in ops.primitives.items()}
        prog = _compile_route("route f = tanh", e)
        arr = np.array(values)
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.tanh(arr))


class TestBinaryOps:
    def test_add(self, env, store):
        prog = _compile_route("route f = add", env)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, a + b)

    def test_multiply(self, env, store):
        prog = _compile_route("route f = multiply", env)
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, a * b)

    def test_subtract(self, env, store):
        prog = _compile_route("route f = subtract", env)
        a = np.array([10.0, 20.0])
        b = np.array([3.0, 7.0])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, a - b)


class TestComposition:
    def test_add_then_tanh(self, env, store):
        prog = _compile_route("route f = add >> tanh", env)
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        out = _run_binary(prog, store, a, b)
        assert np.allclose(out, np.tanh(a + b))

    def test_chain_unary(self, env, store):
        prog = _compile_route("route f = exp >> log", env)
        arr = np.array([1.0, 2.0, 3.0])
        out = _run_unary(prog, store, arr)
        assert np.allclose(out, np.log(np.exp(arr)))

    def test_multiply_then_sqrt(self, env, store):
        prog = _compile_route("route f = multiply >> sqrt", env)
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
        prog = _compile_route("route f = tanh", env)
        for i in range(5):
            arr = np.array([float(i)])
            out = _run_unary(prog, store, arr)
            assert np.allclose(out, np.tanh(arr))
