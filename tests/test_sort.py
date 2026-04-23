"""Sort tests: sort construction, tensor coder, graph assembly, compatibility."""

import numpy as np
import pytest

from unialg import semiring, sort, tensor_coder, build_graph
from unialg.algebra import sort_type_from_term, check_sort_compatibility
from hydra.core import Name, TypeVariable, TypeApplication


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

    def test_sort_type_from_term(self, real_sr):
        h = sort("hidden", real_sr)
        t = sort_type_from_term(h)
        # sort_type_from_term returns a structural TypeApplication, not a TypeVariable
        assert isinstance(t, TypeApplication)
        assert t.value.function == TypeVariable(Name("ua.sort.hidden"))
        assert t.value.argument == TypeVariable(Name("ua.semiring.real"))

    def test_different_sorts_different_types(self, real_sr):
        h = sort("hidden", real_sr)
        o = sort("output", real_sr)
        assert sort_type_from_term(h) != sort_type_from_term(o)

    def test_different_semiring_different_types(self, real_sr, tropical_sr):
        h_real = sort("hidden", real_sr)
        h_trop = sort("hidden", tropical_sr)
        assert sort_type_from_term(h_real) != sort_type_from_term(h_trop)


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
        # Component names are registered, not fused keys
        assert Name("ua.sort.hidden") in g.schema_types
        assert Name("ua.semiring.real") in g.schema_types
        assert Name("ua.sort.output") in g.schema_types

    def test_tensor_type_in_schema(self, real_sr):
        g = build_graph([sort("hidden", real_sr)])
        assert Name("ua.tensor.NDArray") in g.schema_types

