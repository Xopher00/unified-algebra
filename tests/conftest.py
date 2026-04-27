"""Shared test helpers for unified-algebra.

These helpers wrap Hydra's encode/decode conventions and graph-building
boilerplate. They were previously copy-pasted across 20+ test files.
"""

import numpy as np
from hydra.dsl.python import FrozenDict, Right, Left
from hydra.reduction import reduce_term


def encode_array(coder, arr):
    """Encode a numpy array into a Hydra term via the tensor coder."""
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
    assert isinstance(result, Right)
    return result.value


def decode_term(coder, term):
    """Decode a Hydra term back into a numpy array."""
    result = coder.encode(None, None, term)
    assert isinstance(result, Right)
    return result.value


def assert_reduce_ok(cx, graph, term):
    """Reduce a Hydra term and assert success."""
    result = reduce_term(cx, graph, True, term)
    assert isinstance(result, Right), f"reduce_term returned Left: {result}"
    return result.value


def make_graph_with_stdlib(primitives=None, bound_terms=None, sorts=None):
    """Build a Hydra Graph with standard library + extra primitives/bound_terms."""
    from hydra.sources.libraries import standard_library
    from unialg.assembly.graph import build_graph
    all_prims = dict(standard_library())
    if primitives:
        all_prims.update(primitives)
    return build_graph(sorts or [], primitives=all_prims, bound_terms=bound_terms or {})


def build_schema(eq_by_name, extra_sorts=()):
    """Build a FrozenDict schema from resolved equations and optional extra sorts."""
    from unialg.algebra.sort import sort_wrap
    schema = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
    for s in extra_sorts:
        sort_wrap(s).register_schema(schema)
    return FrozenDict(schema)
