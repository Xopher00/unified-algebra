"""Verify the public API is intentional and internal names are not exposed."""
import unialg


_PUBLIC = {
    "Backend", "NumpyBackend", "PytorchBackend", "JaxBackend", "CupyBackend",
    "Semiring", "Equation", "Sort", "ProductSort",
    "Program", "compile_program",
    "parse_ua", "parse_ua_spec", "UASpec",
}

_INTERNAL = {
    "compile_einsum", "semiring_contract", "tensor_coder",
    "contract_and_apply", "contract_merge",
    "rebind_params", "assemble_graph", "build_graph",
    "validate_pipeline", "topo_edges", "type_check_term",
    "NumpyApiBackend",
}


def test_all_contains_expected_public():
    missing = _PUBLIC - set(unialg.__all__)
    assert not missing, f"Missing from __all__: {missing}"


def test_all_has_no_unexpected_extras():
    unexpected = set(unialg.__all__) - _PUBLIC
    assert not unexpected, f"Unexpectedly in __all__: {unexpected}"


def test_internal_names_not_in_all():
    leaked = _INTERNAL & set(unialg.__all__)
    assert not leaked, f"Internal names leaked into __all__: {leaked}"
