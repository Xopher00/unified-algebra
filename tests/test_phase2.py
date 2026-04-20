"""Phase 2 tests: sorts, tensor coder, graph assembly, compatibility."""

import numpy as np
import pytest

from unified_algebra.semiring import semiring
from unified_algebra.sort import (
    sort, sort_to_type, check_sort_compatibility,
    tensor_coder,
)
from unified_algebra.graph import build_graph
from hydra.core import Name, TypeVariable


@pytest.fixture
def real_sr():
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def tropical_sr():
    return semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


class TestSortConstruction:

    def test_sort_has_name(self, real_sr):
        s = sort("hidden", real_sr)
        fields = {f.name.value: f.term for f in s.value.fields}
        assert fields["name"].value.value == "hidden"

    def test_sort_has_semiring(self, real_sr):
        s = sort("hidden", real_sr)
        fields = {f.name.value: f.term for f in s.value.fields}
        sr_fields = {f.name.value: f.term for f in fields["semiring"].value.fields}
        assert sr_fields["name"].value.value == "real"

    def test_different_sorts_different_names(self, real_sr):
        h = sort("hidden", real_sr)
        o = sort("output", real_sr)
        h_name = {f.name.value: f.term for f in h.value.fields}["name"].value.value
        o_name = {f.name.value: f.term for f in o.value.fields}["name"].value.value
        assert h_name != o_name


class TestSortTypeMapping:

    def test_sort_to_type(self):
        t = sort_to_type("hidden", "real")
        assert isinstance(t, TypeVariable)
        assert t.value == Name("ua.sort.hidden:real")

    def test_different_sorts_different_types(self):
        assert sort_to_type("hidden", "real") != sort_to_type("output", "real")

    def test_different_semiring_different_types(self):
        assert sort_to_type("hidden", "real") != sort_to_type("hidden", "tropical")


class TestTensorCoder:

    def test_roundtrip_float64(self):
        coder = tensor_coder()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_float32(self):
        coder = tensor_coder()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_roundtrip_int32(self):
        coder = tensor_coder()
        arr = np.array([1, 2, 3], dtype=np.int32)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_scalar(self):
        coder = tensor_coder()
        arr = np.float64(3.14)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_allclose(result, 3.14)

    def test_roundtrip_3d(self):
        coder = tensor_coder()
        arr = np.arange(24.0).reshape(2, 3, 4)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)

    def test_coder_type(self):
        coder = tensor_coder()
        assert isinstance(coder.type, TypeVariable)
        assert coder.type.value == Name("ua.tensor.NDArray")


class TestCompatibility:

    def test_same_semiring_compatible(self, real_sr):
        h = sort("hidden", real_sr)
        o = sort("output", real_sr)
        assert check_sort_compatibility(h, o) is True

    def test_different_semiring_incompatible(self, real_sr, tropical_sr):
        h = sort("hidden", real_sr)
        t = sort("scores", tropical_sr)
        assert check_sort_compatibility(h, t) is False


class TestGraphAssembly:

    def test_sorts_in_schema(self, real_sr):
        h = sort("hidden", real_sr)
        o = sort("output", real_sr)
        g = build_graph([h, o])
        assert Name("ua.sort.hidden:real") in g.schema_types
        assert Name("ua.sort.output:real") in g.schema_types

    def test_tensor_type_in_schema(self, real_sr):
        g = build_graph([sort("hidden", real_sr)])
        assert Name("ua.tensor.NDArray") in g.schema_types

    def test_sort_terms_in_bound_terms(self, real_sr):
        h = sort("hidden", real_sr)
        g = build_graph([h])
        assert Name("ua.sort.hidden:real") in g.bound_terms
