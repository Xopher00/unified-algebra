"""Test that every ua.* entry point registered in a compiled graph
matches one of the _ENTRY_PREFIXES declared in assembly/program.py.

If a new entry-point naming pattern is added to the assembly layer without
updating _ENTRY_PREFIXES, this test fails and the maintainer is forced to
update the prefix list before the change can land.
"""
from __future__ import annotations

import pytest

from unialg import compile_program, Equation, Semiring, Sort, NumpyBackend
from unialg.assembly.program import _ENTRY_PREFIXES


# Internal prefixes that are allowed to exist in the graph but are NOT entry
# points — they are schema types, parameter bindings, or structural helpers.
_INTERNAL_UA_PREFIXES = (
    "ua.param.",
    "ua.sort.",
    "ua.semiring.",
    "ua.tensor.",
    "ua.lens.",
    "ua.batched",
)


def _all_callable_keys(graph):
    """Return the string names of every primitive and bound_term in *graph*."""
    return [k.value for k in list(graph.primitives) + list(graph.bound_terms)]


def _is_internal(name: str) -> bool:
    return any(name.startswith(p) for p in _INTERNAL_UA_PREFIXES)


def _is_entry_prefix(name: str) -> bool:
    return any(name.startswith(p) for p in _ENTRY_PREFIXES)


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


class TestEntryPrefixCompleteness:

    def test_equation_entry_keys_match_prefixes(self, hidden, real_sr):
        """All ua.* callable keys in an equation-only program match _ENTRY_PREFIXES."""
        backend = NumpyBackend()
        eqs = [
            Equation("linear", "ij,j->i", hidden, hidden, real_sr),
            Equation("relu",   None,       hidden, hidden, nonlinearity="relu"),
            Equation("tanh",   None,       hidden, hidden, nonlinearity="tanh"),
        ]
        prog = compile_program(eqs, backend=backend)
        graph = prog.graph

        for name in _all_callable_keys(graph):
            if not name.startswith("ua."):
                continue
            if _is_internal(name):
                continue
            assert _is_entry_prefix(name), (
                f"Graph key {name!r} starts with 'ua.' but does not match any "
                f"known _ENTRY_PREFIXES: {_ENTRY_PREFIXES}.\n"
                f"Update _ENTRY_PREFIXES in assembly/program.py if this is a "
                f"new intentional entry-point category."
            )

    def test_path_and_fan_entry_keys_match_prefixes(self):
        """All ua.* keys in a program with path/fan cells match _ENTRY_PREFIXES."""
        from unialg import parse_ua
        backend = NumpyBackend()
        prog = parse_ua(
            """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op ep_lin : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
op ep_relu : hidden -> hidden
  nonlinearity = relu
cell ep_path : hidden -> hidden = ep_lin > ep_relu
""",
            backend,
        )
        graph = prog.graph

        for name in _all_callable_keys(graph):
            if not name.startswith("ua."):
                continue
            if _is_internal(name):
                continue
            assert _is_entry_prefix(name), (
                f"Graph key {name!r} starts with 'ua.' but does not match any "
                f"known _ENTRY_PREFIXES: {_ENTRY_PREFIXES}.\n"
                f"Update _ENTRY_PREFIXES in assembly/program.py if this is a "
                f"new intentional entry-point category."
            )

    def test_entry_points_list_covers_all_ua_callables(self, hidden, real_sr):
        """entry_points() returns a name for every ua.* callable that is not internal."""
        backend = NumpyBackend()
        eqs = [
            Equation("ep_lin2", "ij,j->i", hidden, hidden, real_sr),
            Equation("ep_act",  None,       hidden, hidden, nonlinearity="relu"),
        ]
        prog = compile_program(eqs, backend=backend)
        graph = prog.graph

        # Every non-internal ua.* callable must produce a short name via _short_name.
        from unialg.assembly.program import _short_name
        for name in _all_callable_keys(graph):
            if not name.startswith("ua."):
                continue
            if _is_internal(name):
                continue
            short = _short_name(name)
            assert short is not None, (
                f"Graph key {name!r} is a ua.* callable but _short_name returns None. "
                f"Either add its prefix to _ENTRY_PREFIXES or mark it internal."
            )
