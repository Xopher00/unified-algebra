"""Parser tests: .ua text → UASpec → Program.

Covers parse_ua_spec, parse_ua, path, tropical, fan, fold, lens, and error cases.
All 'program runs correctly' tests verify against a numpy oracle.
"""

import numpy as np
import pytest

from unialg import (
    NumpyBackend, parse_ua, parse_ua_spec, UASpec,
    PathSpec, FanSpec, FoldSpec, UnfoldSpec, FixpointSpec, LensPathSpec, LensFanSpec,
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

    def test_returns_uaspec_instance(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert isinstance(spec, UASpec)

    def test_empty_specs_by_default(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""
        spec = parse_ua_spec(text)
        assert spec.specs == []
        assert spec.lenses == []


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
# Test 3: path with >> — compile and run, match numpy oracle
# ---------------------------------------------------------------------------

class TestPathDeclaration:

    def test_path_spec_created(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu

seq layer : hidden -> hidden = linear >> relu
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], PathSpec)
        assert spec.specs[0].name == 'layer'
        assert spec.specs[0].eq_names == ['linear', 'relu']

    def test_path_entry_point_exists(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu

seq layer : hidden -> hidden = linear >> relu
"""
        prog = parse_ua(text, NumpyBackend())
        assert 'layer' in prog.entry_points()

    def test_three_step_path(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op a : hidden -> hidden
  nonlinearity = relu

op b : hidden -> hidden
  nonlinearity = tanh

op c : hidden -> hidden
  nonlinearity = sigmoid

seq abc : hidden -> hidden = a >> b >> c
"""
        spec = parse_ua_spec(text)
        assert spec.specs[0].eq_names == ['a', 'b', 'c']

    def test_path_numpy_oracle(self):
        """seq relu >> tanh should match numpy oracle."""
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

op tanh_eq : hidden -> hidden
  nonlinearity = tanh

seq rt : hidden -> hidden = relu >> tanh_eq
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog('rt', x)
        expected = np.tanh(np.maximum(0.0, x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: tropical semiring — Bellman-Ford 1-hop
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


# ---------------------------------------------------------------------------
# Test 5: fan declaration — parse and check spec type
# ---------------------------------------------------------------------------

class TestFanDeclaration:

    def test_fan_spec_type(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op branch1 : hidden -> hidden
  nonlinearity = relu

op branch2 : hidden -> hidden
  nonlinearity = tanh

op add_merge : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch split : hidden -> hidden = branch1 | branch2
  merge = add_merge
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], FanSpec)

    def test_fan_spec_branch_names(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op a : hidden -> hidden
  nonlinearity = relu

op b : hidden -> hidden
  nonlinearity = tanh

op c : hidden -> hidden
  nonlinearity = sigmoid

op add_merge : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch parallel : hidden -> hidden = a | b | c
  merge = add_merge
"""
        spec = parse_ua_spec(text)
        fan_spec = spec.specs[0]
        assert fan_spec.branch_names == ['a', 'b', 'c']
        assert fan_spec.merge_names == ['add_merge']

    def test_fan_entry_point_exists(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op branch1 : hidden -> hidden
  nonlinearity = relu

op branch2 : hidden -> hidden
  nonlinearity = tanh

op add_merge : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch split : hidden -> hidden = branch1 | branch2
  merge = add_merge
"""
        prog = parse_ua(text, NumpyBackend())
        assert 'split' in prog.entry_points()


# ---------------------------------------------------------------------------
# Test 6: fold declaration — parse and check spec type
# ---------------------------------------------------------------------------

class TestFoldDeclaration:

    def test_fold_spec_type(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op step : hidden -> hidden
  nonlinearity = relu

scan rnn : hidden -> hidden
  step = step
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], FoldSpec)

    def test_fold_spec_step_name(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op my_step : hidden -> hidden
  nonlinearity = tanh

scan rnn : hidden -> hidden
  step = my_step
"""
        spec = parse_ua_spec(text)
        fold_spec = spec.specs[0]
        assert fold_spec.step_name == 'my_step'
        assert fold_spec.name == 'rnn'


# ---------------------------------------------------------------------------
# Test 7: unfold declaration — parse and check spec type
# ---------------------------------------------------------------------------

class TestUnfoldDeclaration:

    def test_unfold_spec_type(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op transition : hidden -> hidden
  nonlinearity = tanh

unroll stream : hidden -> hidden
  step = transition
  steps = 10
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], UnfoldSpec)

    def test_unfold_spec_fields(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op my_step : hidden -> hidden
  nonlinearity = relu

unroll gen : hidden -> hidden
  step = my_step
  steps = 5
"""
        spec = parse_ua_spec(text)
        us = spec.specs[0]
        assert us.name == 'gen'
        assert us.step_name == 'my_step'
        assert us.n_steps == 5


# ---------------------------------------------------------------------------
# Test 8: lens declaration — parse and check lenses list
# ---------------------------------------------------------------------------

class TestLensDeclaration:

    def test_lens_in_lenses_list(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op fwd : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op bwd : hidden -> hidden
  einsum = "ji,j->i"
  algebra = real

lens backprop : hidden <-> hidden
  fwd = fwd
  bwd = bwd
"""
        spec = parse_ua_spec(text)
        assert len(spec.lenses) == 1

    def test_multiple_lenses(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op a_fwd : hidden -> hidden
  nonlinearity = relu

op a_bwd : hidden -> hidden
  nonlinearity = relu

op b_fwd : hidden -> hidden
  nonlinearity = tanh

op b_bwd : hidden -> hidden
  nonlinearity = tanh

lens lens_a : hidden <-> hidden
  fwd = a_fwd
  bwd = a_bwd

lens lens_b : hidden <-> hidden
  fwd = b_fwd
  bwd = b_bwd
"""
        spec = parse_ua_spec(text)
        assert len(spec.lenses) == 2


# ---------------------------------------------------------------------------
# Test 8: error — unknown sort name raises ValueError
# ---------------------------------------------------------------------------

class TestUnknownSortError:

    def test_unknown_sort_in_equation(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op bad : unknown -> hidden
  nonlinearity = relu
"""
        with pytest.raises(ValueError, match="Unknown spec 'unknown'"):
            parse_ua_spec(text)

    def test_error_message_lists_known_sorts(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec visible(real)

op bad : missing_sort -> visible
  nonlinearity = relu
"""
        with pytest.raises(ValueError, match="visible"):
            parse_ua_spec(text)

    def test_unknown_equation_in_path(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

seq bad_path : hidden -> hidden = nonexistent_eq
"""
        with pytest.raises(ValueError, match="op 'nonexistent_eq' not found"):
            parse_ua(text, NumpyBackend())

    def test_unknown_semiring_in_sort(self):
        text = """
spec hidden(nonexistent_semiring)
"""
        with pytest.raises(ValueError, match="Unknown algebra 'nonexistent_semiring'"):
            parse_ua_spec(text)


# ---------------------------------------------------------------------------
# Test 9: bad syntax raises SyntaxError
# ---------------------------------------------------------------------------

class TestSyntaxErrors:

    def test_missing_closing_paren(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)

    def test_unrecognised_keyword_at_top_level(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
unknown_keyword foo
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)

    def test_missing_arrow_in_equation(self):
        """op without -> in signature should raise SyntaxError."""
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op bad : hidden hidden
  nonlinearity = relu
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)


# ---------------------------------------------------------------------------
# Test 10: comments are ignored everywhere
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
# Test: fixpoint declaration — parse and check spec type and fields
# ---------------------------------------------------------------------------

class TestFixpointDeclaration:

    def test_fixpoint_spec_type(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op step_eq : hidden -> hidden
  nonlinearity = tanh

op pred_eq : hidden -> hidden
  nonlinearity = sigmoid

fixpoint converge : hidden
  step = step_eq
  predicate = pred_eq
  epsilon = 0.001
  max_iter = 50
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], FixpointSpec)

    def test_fixpoint_spec_fields(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op step_eq : hidden -> hidden
  nonlinearity = tanh

op pred_eq : hidden -> hidden
  nonlinearity = sigmoid

fixpoint converge : hidden
  step = step_eq
  predicate = pred_eq
  epsilon = 0.01
  max_iter = 200
"""
        spec = parse_ua_spec(text)
        fp = spec.specs[0]
        assert fp.name == 'converge'
        assert fp.step_name == 'step_eq'
        assert fp.predicate_name == 'pred_eq'
        assert fp.epsilon == 0.01
        assert fp.max_iter == 200


# ---------------------------------------------------------------------------
# Integration: full program parsing and execution
# ---------------------------------------------------------------------------

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

    def test_fan_runs_correctly(self):
        """Branch: relu and tanh branches, Hadamard-product merge.

        "i,i->i" with real algebra (times=multiply) computes relu(x) * tanh(x).
        No contracted indices — pure elementwise product of branch outputs.
        """
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op branch_relu : hidden -> hidden
  nonlinearity = relu

op branch_tanh : hidden -> hidden
  nonlinearity = tanh

op hadamard_merge : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch dual : hidden -> hidden = branch_relu | branch_tanh
  merge = hadamard_merge
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([1.0, -1.0, 0.5])
        out = prog('dual', x)
        # "i,i->i" with times=multiply: elementwise product of branch outputs
        expected = np.maximum(0, x) * np.tanh(x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Residual (skip connection) paths
# ---------------------------------------------------------------------------

class TestResidualPath:

    def test_residual_path_spec(self):
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu

seq layer+ : hidden -> hidden = linear >> relu
  algebra = real
"""
        spec = parse_ua_spec(text)
        ps = spec.specs[0]
        assert isinstance(ps, PathSpec)
        assert ps.residual is True
        assert ps.residual_semiring == 'real'

    def test_residual_path_numpy_oracle(self):
        """residual seq: output = relu(x) + x"""
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

seq skip+ : hidden -> hidden = relu
  algebra = real
"""
        prog = parse_ua(text, NumpyBackend())
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog('skip', x)
        expected = np.maximum(0.0, x) + x  # relu(x) + x
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_path_without_residual_unchanged(self):
        """Non-residual seqs still work."""
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu

seq simple : hidden -> hidden = relu
"""
        spec = parse_ua_spec(text)
        ps = spec.specs[0]
        assert ps.residual is False


# ---------------------------------------------------------------------------
# Test: lens_fan declaration — parse, check spec, and error cases
# ---------------------------------------------------------------------------

_LENS_FAN_BASE = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op fwd1 : hidden -> hidden
  nonlinearity = relu

op bwd1 : hidden -> hidden
  nonlinearity = relu

op fwd2 : hidden -> hidden
  nonlinearity = tanh

op bwd2 : hidden -> hidden
  nonlinearity = tanh

op merge_fwd : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

op merge_bwd : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

lens backprop1 : hidden <-> hidden
  fwd = fwd1
  bwd = bwd1

lens backprop2 : hidden <-> hidden
  fwd = fwd2
  bwd = bwd2

lens merge_lens : hidden <-> hidden
  fwd = merge_fwd
  bwd = merge_bwd
"""


class TestLensFanDeclaration:

    def test_lens_fan_parse(self):
        text = _LENS_FAN_BASE + """
lens_branch attention : hidden <-> hidden = backprop1 | backprop2
  merge = merge_lens
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        lfs = spec.specs[0]
        assert isinstance(lfs, LensFanSpec)
        assert lfs.name == 'attention'
        assert lfs.branch_names == ['fwd1', 'fwd2']
        assert lfs.merge_name == 'merge_fwd'
        assert lfs.domain_sort is not None
        assert lfs.codomain_sort is not None

    def test_lens_fan_unknown_lens_error(self):
        text = _LENS_FAN_BASE + """
lens_branch attention : hidden <-> hidden = backprop1 | nonexistent_lens
  merge = merge_lens
"""
        with pytest.raises(ValueError, match="Unknown lens 'nonexistent_lens'"):
            parse_ua_spec(text)

    def test_lens_fan_unknown_merge_error(self):
        text = _LENS_FAN_BASE + """
lens_branch attention : hidden <-> hidden = backprop1 | backprop2
  merge = nonexistent_merge
"""
        with pytest.raises(ValueError, match="Unknown lens 'nonexistent_merge'"):
            parse_ua_spec(text)


# ---------------------------------------------------------------------------
# Test: template equations (Para instantiation)
# ---------------------------------------------------------------------------

_TEMPLATE_BASE = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
"""

_PROJ_TEMPLATE = _TEMPLATE_BASE + """
op ~proj : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
"""


class TestTemplateEquations:

    def test_template_equation_not_in_equations(self):
        """An op with ~ prefix must NOT appear in spec.equations."""
        spec = parse_ua_spec(_PROJ_TEMPLATE)
        assert len(spec.equations) == 0

    def test_template_expansion_in_fan(self):
        """Branch with proj[q], proj[k], proj[v] should expand to 3 concrete equations."""
        text = _PROJ_TEMPLATE + """
op merge : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

branch kv : hidden -> hidden = proj[q] | proj[k] | proj[v]
  merge = merge
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [eq.name for eq in spec.equations]
        assert 'q_proj' in eq_names_in_spec
        assert 'k_proj' in eq_names_in_spec
        assert 'v_proj' in eq_names_in_spec
        fan_spec = spec.specs[0]
        assert fan_spec.branch_names == ['q_proj', 'k_proj', 'v_proj']

    def test_template_expansion_in_path(self):
        """Seq with proj[q] >> proj[k] should expand to two concrete equations."""
        text = _PROJ_TEMPLATE + """
seq qk : hidden -> hidden = proj[q] >> proj[k]
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [eq.name for eq in spec.equations]
        assert 'q_proj' in eq_names_in_spec
        assert 'k_proj' in eq_names_in_spec
        path_spec = spec.specs[0]
        assert path_spec.eq_names == ['q_proj', 'k_proj']

    def test_template_reuse_same_prefix(self):
        """Using proj[q] twice in the same seq must produce exactly ONE q_proj equation."""
        text = _PROJ_TEMPLATE + """
seq qq : hidden -> hidden = proj[q] >> proj[q]
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [eq.name for eq in spec.equations]
        q_proj_count = eq_names_in_spec.count('q_proj')
        assert q_proj_count == 1

    def test_template_unknown_raises(self):
        """Referencing an undeclared template name should raise ValueError."""
        text = _TEMPLATE_BASE + """
seq bad : hidden -> hidden = unknown[x]
"""
        with pytest.raises(ValueError, match="Unknown template 'unknown'"):
            parse_ua_spec(text)

    def test_template_mixed_refs(self):
        """Mix of plain idents and template refs in a seq should both resolve."""
        text = _TEMPLATE_BASE + """
op ~proj : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu

seq qr : hidden -> hidden = proj[q] >> relu
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [eq.name for eq in spec.equations]
        assert 'q_proj' in eq_names_in_spec
        assert 'relu' in eq_names_in_spec
        path_spec = spec.specs[0]
        assert path_spec.eq_names == ['q_proj', 'relu']
