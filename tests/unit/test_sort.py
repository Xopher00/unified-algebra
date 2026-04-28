"""Sort tests: sort construction, tensor coder, graph assembly, compatibility."""

import numpy as np
import pytest

from unialg import Semiring, Sort
from unialg.terms import tensor_coder
from unialg.assembly.graph import build_graph
from unialg.algebra.sort import check_sort_compatibility
from hydra.core import Name, TypeVariable, TypeApplication


class TestSortConstruction:

    def test_sort_has_name(self, real_sr):
        s = Sort("hidden", real_sr)
        fields = {f.name.value: f.term for f in s.term.value.fields}
        assert fields["name"].value.value == "hidden"

    def test_sort_has_semiring(self, real_sr):
        s = Sort("hidden", real_sr)
        fields = {f.name.value: f.term for f in s.term.value.fields}
        sr_fields = {f.name.value: f.term for f in fields["semiring"].value.fields}
        assert sr_fields["name"].value.value == "real"

    def test_different_sorts_different_names(self, real_sr):
        h = Sort("hidden", real_sr)
        o = Sort("output", real_sr)
        h_name = {f.name.value: f.term for f in h.term.value.fields}["name"].value.value
        o_name = {f.name.value: f.term for f in o.term.value.fields}["name"].value.value
        assert h_name != o_name


class TestSortTypeMapping:

    def test_sort_type_from_term(self, real_sr):
        h = Sort("hidden", real_sr)
        t = Sort.from_term(h).type_
        # Sort.from_term(h).type_ returns a structural TypeApplication, not a TypeVariable
        assert isinstance(t, TypeApplication)
        assert t.value.function == TypeVariable(Name("ua.sort.hidden"))
        assert t.value.argument == TypeVariable(Name("ua.semiring.real"))

    def test_different_sorts_different_types(self, real_sr):
        h = Sort("hidden", real_sr)
        o = Sort("output", real_sr)
        assert Sort.from_term(h).type_ != Sort.from_term(o).type_

    def test_different_semiring_different_types(self, real_sr, tropical_sr):
        h_real = Sort("hidden", real_sr)
        h_trop = Sort("hidden", tropical_sr)
        assert Sort.from_term(h_real).type_ != Sort.from_term(h_trop).type_


class TestTensorCoder:

    def test_roundtrip_float64(self, backend):
        coder = tensor_coder(backend)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_float32(self, backend):
        coder = tensor_coder(backend)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_roundtrip_int32(self, backend):
        coder = tensor_coder(backend)
        arr = np.array([1, 2, 3], dtype=np.int32)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_scalar(self, backend):
        coder = tensor_coder(backend)
        arr = np.float64(3.14)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_allclose(result, 3.14)

    def test_roundtrip_3d(self, backend):
        coder = tensor_coder(backend)
        arr = np.arange(24.0).reshape(2, 3, 4)
        term = coder.decode(None, arr).value
        result = coder.encode(None, None, term).value
        np.testing.assert_array_equal(result, arr)

    def test_coder_type(self, backend):
        coder = tensor_coder(backend)
        assert isinstance(coder.type, TypeVariable)
        assert coder.type.value == Name("ua.tensor.NDArray")


class TestCompatibility:

    def test_same_semiring_compatible(self, real_sr):
        h = Sort("hidden", real_sr)
        o = Sort("output", real_sr)
        assert check_sort_compatibility(h, o) is True

    def test_different_semiring_incompatible(self, real_sr, tropical_sr):
        h = Sort("hidden", real_sr)
        t = Sort("scores", tropical_sr)
        assert check_sort_compatibility(h, t) is False


class TestGraphAssembly:

    def test_sorts_in_schema(self, real_sr):
        h = Sort("hidden", real_sr)
        o = Sort("output", real_sr)
        g = build_graph([h, o])
        # Component names are registered, not fused keys
        assert Name("ua.sort.hidden") in g.schema_types
        assert Name("ua.semiring.real") in g.schema_types
        assert Name("ua.sort.output") in g.schema_types

    def test_tensor_type_in_schema(self, real_sr):
        g = build_graph([Sort("hidden", real_sr)])
        assert Name("ua.tensor.NDArray") in g.schema_types


class TestSizedAxes:

    @pytest.fixture
    def real_sr(self):
        return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)

    def test_axis_names_from_unsized(self, real_sr):
        s = Sort("h", real_sr, axes=("batch", "feature"))
        assert s.axis_names == ["batch", "feature"]
        assert s.axis_dims == [None, None]

    def test_axis_names_from_sized(self, real_sr):
        s = Sort("h", real_sr, axes=("batch", "feature:128"))
        assert s.axis_names == ["batch", "feature"]
        assert s.axis_dims == [None, 128]

    def test_axis_dims_all_sized(self, real_sr):
        s = Sort("h", real_sr, axes=("batch:32", "feature:128"))
        assert s.axis_names == ["batch", "feature"]
        assert s.axis_dims == [32, 128]

    def test_rank_unchanged_by_sizes(self, real_sr):
        s = Sort("h", real_sr, axes=("batch:32", "feature:128"))
        assert s.rank == 2

    def test_no_axes_gives_empty(self, real_sr):
        s = Sort("h", real_sr)
        assert s.axis_names == []
        assert s.axis_dims == []

