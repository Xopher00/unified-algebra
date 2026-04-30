"""Parser tests: .ua text → UASpec → Program.

Covers parse_ua_spec, parse_ua, path, tropical, fan, fold, lens, fixpoint,
template equations, residual paths, and integration tests.
All 'program runs correctly' tests verify against a numpy oracle.
"""

import numpy as np
import pytest

from unialg import (
    NumpyBackend, ProductSort, parse_ua, parse_ua_spec, UASpec,
)
from unialg import Equation


# ---------------------------------------------------------------------------
# Test 1: parse_ua_spec on semiring + sort + equation triple
# ---------------------------------------------------------------------------

class TestParseUASpec:

    def test_semiring_populated(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
"""
        spec = parse_ua_spec(text)
        assert 'real' in spec.semirings

    def test_sort_populated(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
spec visible(real)
"""
        spec = parse_ua_spec(text)
        assert 'hidden' in spec.sorts
        assert 'visible' in spec.sorts

    def test_equation_in_list(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu
"""
        spec = parse_ua_spec(text)
        assert len(spec.equations) == 2

    def test_product_domain_in_sort_signature(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec left(real)
spec right(real)
spec output(real)

op combine : (left, right) -> output
  einsum = "i,j->ij"
  algebra = real
"""
        spec = parse_ua_spec(text)
        eq = spec.equations[0]
        assert isinstance(eq.domain_sort, ProductSort)
        assert eq.codomain_sort.name == "output"

    def test_returns_uaspec_instance(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert isinstance(spec, UASpec)

    def test_empty_cells_by_default(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert spec.cells == []


# ---------------------------------------------------------------------------
# Test 2: parse_ua produces a callable Program, invoke and check output
# ---------------------------------------------------------------------------

class TestParseUA:

    def test_returns_callable_program(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        out = prog('relu', x)
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_linear_equation_result(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
"""
        prog = parse_ua(text, NumpyBackend())
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 0.0])
        out = prog('linear', W, x)
        np.testing.assert_allclose(out, W @ x)

    def test_entry_points_include_equations(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu
"""
        prog = parse_ua(text, NumpyBackend())
        eps = prog.entry_points()
        assert 'linear' in eps
        assert 'relu' in eps


# ---------------------------------------------------------------------------
# Test 3: comments are ignored everywhere
# ---------------------------------------------------------------------------

class TestComments:

    def test_leading_comment_ignored(self):
        text = """# top-level comment
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
# another comment
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert 'real' in spec.semirings
        assert 'hidden' in spec.sorts

    def test_inline_comment_ignored(self):
        text = """algebra real(plus=add, times=multiply, zero=0.0, one=1.0) # this is the real semiring
spec hidden(real) # hidden state sort
"""
        spec = parse_ua_spec(text)
        assert 'real' in spec.semirings
        assert 'hidden' in spec.sorts

    def test_blank_lines_between_declarations(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)


spec hidden(real)


op relu : hidden -> hidden
  nonlinearity = relu

"""
        spec = parse_ua_spec(text)
        assert len(spec.equations) == 1

    def test_comment_only_file_gives_empty_spec(self):
        text = """
# just comments
# nothing here
"""
        spec = parse_ua_spec(text)
        assert spec.semirings == {}
        assert spec.sorts == {}
        assert spec.equations == []


# ---------------------------------------------------------------------------
# Test 4: template equations (Para instantiation)
# ---------------------------------------------------------------------------

_TEMPLATE_BASE = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""

_INPUTS_BASE = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""


class TestOpInputsAttribute:

    def test_single_input_parsed(self):
        """inputs = linear sets the equation's inputs tuple to ('linear',)."""
        spec = parse_ua_spec(_INPUTS_BASE + """
op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu
  inputs = linear
""")
        relu_eq = next(eq for eq in spec.equations if eq.name == 'relu')
        assert tuple(relu_eq.inputs) == ('linear',)

    def test_multi_input_parsed(self):
        """inputs = a, b sets the equation's inputs tuple to ('a', 'b')."""
        spec = parse_ua_spec(_INPUTS_BASE + """
op a : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op b : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op c : hidden -> hidden
  einsum = "ij,jk->ik"
  algebra = real
  inputs = a, b
""")
        c_eq = next(eq for eq in spec.equations if eq.name == 'c')
        assert tuple(c_eq.inputs) == ('a', 'b')

    def test_no_inputs_defaults_empty(self):
        """An op without inputs = has an empty inputs tuple."""
        spec = parse_ua_spec(_INPUTS_BASE + """
op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
""")
        linear_eq = spec.equations[0]
        assert tuple(linear_eq.inputs) == ()

    def test_inputs_does_not_affect_other_attrs(self):
        """inputs attribute coexists with einsum and algebra attributes."""
        spec = parse_ua_spec(_INPUTS_BASE + """
op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op downstream : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
  inputs = linear
""")
        ds_eq = next(eq for eq in spec.equations if eq.name == 'downstream')
        assert ds_eq.einsum == "ij,j->i"
        assert ds_eq.semiring_name == "real"
        assert tuple(ds_eq.inputs) == ('linear',)

    def test_string_attributes_support_json_style_escapes(self):
        spec = parse_ua_spec(_INPUTS_BASE + """
op escaped : hidden -> hidden
  einsum = "i,\\"j\\"\\\\k\\n->i"
  algebra = real
""")
        escaped_eq = spec.equations[0]
        assert escaped_eq.einsum == 'i,"j"\\k\n->i'

    def test_string_attributes_reject_raw_newlines(self):
        with pytest.raises(SyntaxError):
            parse_ua_spec(_INPUTS_BASE + """
op bad : hidden -> hidden
  einsum = "i,
->i"
  algebra = real
""")


# ---------------------------------------------------------------------------
# share declaration
# ---------------------------------------------------------------------------

_SHARE_BASE = """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op q_proj : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op k_proj : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op v_proj : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

seq attn_q : hidden -> hidden = q_proj
seq attn_k : hidden -> hidden = k_proj
seq attn_v : hidden -> hidden = v_proj
"""


class TestIntegration:

    def test_two_semirings_two_sorts(self):
        """Parser handles multiple algebras and specs in one file."""
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec hidden(real)
spec node(tropical)
"""
        spec = parse_ua_spec(text)
        assert len(spec.semirings) == 2
        assert len(spec.sorts) == 2

    def test_batched_sort(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden_batched(real, batched)
"""
        spec = parse_ua_spec(text)
        assert 'hidden_batched' in spec.sorts
        from unialg.algebra.sort import Sort
        assert Sort.from_term(spec.sorts['hidden_batched']).batched

    def test_equation_with_nonlinearity_only(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op sigmoid_eq : hidden -> hidden
  nonlinearity = sigmoid
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([-1.0, 0.0, 1.0])
        out = prog('sigmoid_eq', x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_sized_axes_parse(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real, axes=[batch, feature:128])
"""
        spec = parse_ua_spec(text)
        from unialg.algebra.sort import Sort
        s = Sort.from_term(spec.sorts['hidden'])
        assert s.axes == ['batch', 'feature:128']
        assert s.axis_names == ['batch', 'feature']
        assert s.axis_dims == [None, 128]

    def test_all_sized_axes_parse(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real, axes=[batch:32, feature:128])
"""
        spec = parse_ua_spec(text)
        from unialg.algebra.sort import Sort
        s = Sort.from_term(spec.sorts['hidden'])
        assert s.axis_dims == [32, 128]
        assert s.rank == 2


# ---------------------------------------------------------------------------
# Path with >> — compile and run, match numpy oracle
# ---------------------------------------------------------------------------

class TestTropicalSemiring:

    def test_tropical_semiring_parsed(self):
        text = """
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec node(tropical)
"""
        spec = parse_ua_spec(text)
        assert 'tropical' in spec.semirings

    def test_bellman_ford_one_hop(self):
        """Tropical einsum ij,j->i implements min_j(W_ij + x_j)."""
        text = """
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec node(tropical)

op sp : node -> node
  einsum = "ij,j->i"
  algebra = tropical
"""
        prog = parse_ua(text, NumpyBackend())
        inf = float('inf')
        W = np.array([[inf, inf, inf],
                      [2.0, inf, inf],
                      [5.0, 1.0, inf]], dtype=float)
        x = np.array([0.0, inf, inf], dtype=float)
        out = prog('sp', W, x)
        np.testing.assert_allclose(out, np.array([inf, 2.0, 5.0]))

    def test_tropical_negative_inf_in_literal(self):
        """Parser handles -inf in algebra declaration."""
        text = """
algebra max_plus(plus=maximum, times=add, zero=-inf, one=0.0)
spec node(max_plus)
"""
        spec = parse_ua_spec(text)
        assert 'max_plus' in spec.semirings

    def test_negative_finite_algebra_literals(self):
        text = """
algebra shifted(plus=add, times=multiply, zero=-1.0, one=-2)
spec hidden(shifted)
"""
        spec = parse_ua_spec(text)
        sr = spec.semirings["shifted"]
        assert sr.zero == -1.0
        assert sr.one == -2.0

    def test_negative_finite_cell_literal(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
cell bias : hidden -> hidden = -1.0
"""
        spec = parse_ua_spec(text)
        assert len(spec.cells) == 1


# ---------------------------------------------------------------------------
# Fan declaration — parse and check spec type
# ---------------------------------------------------------------------------

class TestDefineDeclaration:

    def test_define_unary_parses(self):
        text = """
define unary clamp(x) = minimum(1.0, maximum(0.0, x))
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert len(spec.defines) == 1
        arity, name, params, body = spec.defines[0]
        assert arity == 'unary'
        assert name == 'clamp'
        assert params == ['x']
        assert body[0] == 'call'
        assert body[1] == 'minimum'

    def test_define_binary_parses(self):
        text = """
define binary smooth_max(a, b) = log(exp(a) + exp(b))
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert len(spec.defines) == 1
        arity, name, params, body = spec.defines[0]
        assert arity == 'binary'
        assert name == 'smooth_max'
        assert params == ['a', 'b']
        # Body should be: call('log', [call('add', [call('exp', [var('a')]), call('exp', [var('b')])])])
        assert body == ('call', 'log', [('call', 'add', [('call', 'exp', [('var', 'a')]), ('call', 'exp', [('var', 'b')])])])

    def test_define_infix_precedence(self):
        text = """
define binary f(a, b) = a + b * 2.0
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        _, _, _, body = spec.defines[0]
        # * binds tighter than +: add(var(a), multiply(var(b), lit(2.0)))
        assert body == ('call', 'add', [('var', 'a'), ('call', 'multiply', [('var', 'b'), ('lit', 2.0)])])

    def test_define_unary_minus(self):
        text = """
define unary neg_relu(x) = -maximum(0.0, x)
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        _, _, _, body = spec.defines[0]
        assert body[0] == 'call'
        assert body[1] == 'neg'

    def test_define_negative_literal_uses_unary_minus(self):
        text = """
define unary shift(x) = x + -1.0
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        _, _, _, body = spec.defines[0]
        assert body == ('call', 'add', [
            ('var', 'x'),
            ('call', 'neg', [('lit', 1.0)]),
        ])

    def test_multiple_defines(self):
        text = """
define unary act(x) = tanh(x)
define binary my_plus(a, b) = maximum(a, b)
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert len(spec.defines) == 2
        assert spec.defines[0][1] == 'act'
        assert spec.defines[1][1] == 'my_plus'
