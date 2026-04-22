"""Parser tests: .ua text → UASpec → Program.

Covers parse_ua_spec, parse_ua, path, tropical, fan, fold, lens, and error cases.
All 'program runs correctly' tests verify against a numpy oracle.
"""

import numpy as np
import pytest

from unialg import (
    numpy_backend, parse_ua, parse_ua_spec, UASpec,
    PathSpec, FanSpec, FoldSpec, UnfoldSpec, FixpointSpec, LensPathSpec, LensFanSpec,
)
from unialg.views import EquationView


# ---------------------------------------------------------------------------
# Test 1: parse_ua_spec on semiring + sort + equation triple
# ---------------------------------------------------------------------------

class TestParseUASpec:

    def test_semiring_populated(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real
"""
        spec = parse_ua_spec(text)
        assert 'real' in spec.semirings

    def test_sort_populated(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)
sort visible(real)
"""
        spec = parse_ua_spec(text)
        assert 'hidden' in spec.sorts
        assert 'visible' in spec.sorts

    def test_equation_in_list(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation relu : hidden -> hidden
  nonlinearity = relu
"""
        spec = parse_ua_spec(text)
        assert len(spec.equations) == 2

    def test_returns_uaspec_instance(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)
"""
        spec = parse_ua_spec(text)
        assert isinstance(spec, UASpec)

    def test_empty_specs_by_default(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation relu : hidden -> hidden
  nonlinearity = relu
"""
        prog = parse_ua(text, numpy_backend())
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        out = prog('relu', x)
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_linear_equation_result(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real
"""
        prog = parse_ua(text, numpy_backend())
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 0.0])
        out = prog('linear', W, x)
        np.testing.assert_allclose(out, W @ x)

    def test_entry_points_include_equations(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation relu : hidden -> hidden
  nonlinearity = relu
"""
        prog = parse_ua(text, numpy_backend())
        eps = prog.entry_points()
        assert 'linear' in eps
        assert 'relu' in eps


# ---------------------------------------------------------------------------
# Test 3: path with >> — compile and run, match numpy oracle
# ---------------------------------------------------------------------------

class TestPathDeclaration:

    def test_path_spec_created(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation relu : hidden -> hidden
  nonlinearity = relu

path layer : hidden -> hidden = linear >> relu
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], PathSpec)
        assert spec.specs[0].name == 'layer'
        assert spec.specs[0].eq_names == ['linear', 'relu']

    def test_path_entry_point_exists(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation relu : hidden -> hidden
  nonlinearity = relu

path layer : hidden -> hidden = linear >> relu
"""
        prog = parse_ua(text, numpy_backend())
        assert 'layer' in prog.entry_points()

    def test_three_step_path(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation a : hidden -> hidden
  nonlinearity = relu

equation b : hidden -> hidden
  nonlinearity = tanh

equation c : hidden -> hidden
  nonlinearity = sigmoid

path abc : hidden -> hidden = a >> b >> c
"""
        spec = parse_ua_spec(text)
        assert spec.specs[0].eq_names == ['a', 'b', 'c']

    def test_path_numpy_oracle(self):
        """path relu >> tanh should match numpy oracle."""
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation relu : hidden -> hidden
  nonlinearity = relu

equation tanh_eq : hidden -> hidden
  nonlinearity = tanh

path rt : hidden -> hidden = relu >> tanh_eq
"""
        prog = parse_ua(text, numpy_backend())
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
semiring tropical(plus=minimum, times=add, zero=inf, one=0.0)
sort node(tropical)
"""
        spec = parse_ua_spec(text)
        assert 'tropical' in spec.semirings

    def test_bellman_ford_one_hop(self):
        """Tropical einsum ij,j->i implements min_j(W_ij + x_j)."""
        text = """
semiring tropical(plus=minimum, times=add, zero=inf, one=0.0)
sort node(tropical)

equation sp : node -> node
  einsum = "ij,j->i"
  semiring = tropical
"""
        prog = parse_ua(text, numpy_backend())
        inf = float('inf')
        W = np.array([[inf, inf, inf],
                      [2.0, inf, inf],
                      [5.0, 1.0, inf]], dtype=float)
        x = np.array([0.0, inf, inf], dtype=float)
        out = prog('sp', W, x)
        np.testing.assert_allclose(out, np.array([inf, 2.0, 5.0]))

    def test_tropical_negative_inf_in_literal(self):
        """Parser handles -inf in semiring declaration."""
        text = """
semiring max_plus(plus=maximum, times=add, zero=-inf, one=0.0)
sort node(max_plus)
"""
        spec = parse_ua_spec(text)
        assert 'max_plus' in spec.semirings


# ---------------------------------------------------------------------------
# Test 5: fan declaration — parse and check spec type
# ---------------------------------------------------------------------------

class TestFanDeclaration:

    def test_fan_spec_type(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation branch1 : hidden -> hidden
  nonlinearity = relu

equation branch2 : hidden -> hidden
  nonlinearity = tanh

equation add_merge : hidden -> hidden
  einsum = "i,i->i"
  semiring = real

fan split : hidden -> hidden
  branches = [branch1, branch2]
  merge = add_merge
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], FanSpec)

    def test_fan_spec_branch_names(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation a : hidden -> hidden
  nonlinearity = relu

equation b : hidden -> hidden
  nonlinearity = tanh

equation c : hidden -> hidden
  nonlinearity = sigmoid

equation add_merge : hidden -> hidden
  einsum = "i,i->i"
  semiring = real

fan parallel : hidden -> hidden
  branches = [a, b, c]
  merge = add_merge
"""
        spec = parse_ua_spec(text)
        fan_spec = spec.specs[0]
        assert fan_spec.branch_names == ['a', 'b', 'c']
        assert fan_spec.merge_name == 'add_merge'

    def test_fan_entry_point_exists(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation branch1 : hidden -> hidden
  nonlinearity = relu

equation branch2 : hidden -> hidden
  nonlinearity = tanh

equation add_merge : hidden -> hidden
  einsum = "i,i->i"
  semiring = real

fan split : hidden -> hidden
  branches = [branch1, branch2]
  merge = add_merge
"""
        prog = parse_ua(text, numpy_backend())
        assert 'split' in prog.entry_points()


# ---------------------------------------------------------------------------
# Test 6: fold declaration — parse and check spec type
# ---------------------------------------------------------------------------

class TestFoldDeclaration:

    def test_fold_spec_type(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation step : hidden -> hidden
  nonlinearity = relu

fold rnn : hidden -> hidden
  step = step
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], FoldSpec)

    def test_fold_spec_step_name(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation my_step : hidden -> hidden
  nonlinearity = tanh

fold rnn : hidden -> hidden
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation transition : hidden -> hidden
  nonlinearity = tanh

unfold stream : hidden -> hidden
  step = transition
  n_steps = 10
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        assert isinstance(spec.specs[0], UnfoldSpec)

    def test_unfold_spec_fields(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation my_step : hidden -> hidden
  nonlinearity = relu

unfold gen : hidden -> hidden
  step = my_step
  n_steps = 5
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation fwd : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation bwd : hidden -> hidden
  einsum = "ji,j->i"
  semiring = real

lens backprop : hidden <-> hidden
  fwd = fwd
  bwd = bwd
"""
        spec = parse_ua_spec(text)
        assert len(spec.lenses) == 1

    def test_multiple_lenses(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation a_fwd : hidden -> hidden
  nonlinearity = relu

equation a_bwd : hidden -> hidden
  nonlinearity = relu

equation b_fwd : hidden -> hidden
  nonlinearity = tanh

equation b_bwd : hidden -> hidden
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation bad : unknown -> hidden
  nonlinearity = relu
"""
        with pytest.raises(ValueError, match="Unknown sort 'unknown'"):
            parse_ua_spec(text)

    def test_error_message_lists_known_sorts(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort visible(real)

equation bad : missing_sort -> visible
  nonlinearity = relu
"""
        with pytest.raises(ValueError, match="visible"):
            parse_ua_spec(text)

    def test_unknown_equation_in_path(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

path bad_path : hidden -> hidden = nonexistent_eq
"""
        with pytest.raises(ValueError, match="Unknown equation 'nonexistent_eq'"):
            parse_ua_spec(text)

    def test_unknown_semiring_in_sort(self):
        text = """
sort hidden(nonexistent_semiring)
"""
        with pytest.raises(ValueError, match="Unknown semiring 'nonexistent_semiring'"):
            parse_ua_spec(text)


# ---------------------------------------------------------------------------
# Test 9: bad syntax raises SyntaxError
# ---------------------------------------------------------------------------

class TestSyntaxErrors:

    def test_missing_closing_paren(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)

    def test_unrecognised_keyword_at_top_level(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)
unknown_keyword foo
"""
        with pytest.raises(SyntaxError):
            parse_ua_spec(text)

    def test_missing_arrow_in_equation(self):
        """equation without -> in signature should raise SyntaxError."""
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)
equation bad : hidden hidden
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
# another comment
sort hidden(real)
"""
        spec = parse_ua_spec(text)
        assert 'real' in spec.semirings
        assert 'hidden' in spec.sorts

    def test_inline_comment_ignored(self):
        text = """semiring real(plus=add, times=multiply, zero=0.0, one=1.0) # this is the real semiring
sort hidden(real) # hidden state sort
"""
        spec = parse_ua_spec(text)
        assert 'real' in spec.semirings
        assert 'hidden' in spec.sorts

    def test_blank_lines_between_declarations(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)


sort hidden(real)


equation relu : hidden -> hidden
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation step_eq : hidden -> hidden
  nonlinearity = tanh

equation pred_eq : hidden -> hidden
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation step_eq : hidden -> hidden
  nonlinearity = tanh

equation pred_eq : hidden -> hidden
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
        """Parser handles multiple semirings and sorts in one file."""
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
semiring tropical(plus=minimum, times=add, zero=inf, one=0.0)
sort hidden(real)
sort node(tropical)
"""
        spec = parse_ua_spec(text)
        assert len(spec.semirings) == 2
        assert len(spec.sorts) == 2

    def test_batched_sort(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden_batched(real, batched)
"""
        spec = parse_ua_spec(text)
        assert 'hidden_batched' in spec.sorts
        from unialg import is_batched
        assert is_batched(spec.sorts['hidden_batched'])

    def test_equation_with_nonlinearity_only(self):
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation sigmoid_eq : hidden -> hidden
  nonlinearity = sigmoid
"""
        prog = parse_ua(text, numpy_backend())
        x = np.array([-1.0, 0.0, 1.0])
        out = prog('sigmoid_eq', x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_fan_runs_correctly(self):
        """Fan: relu and tanh branches, Hadamard-product merge.

        "i,i->i" with real semiring (times=multiply) computes relu(x) * tanh(x).
        No contracted indices — pure elementwise product of branch outputs.
        """
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation branch_relu : hidden -> hidden
  nonlinearity = relu

equation branch_tanh : hidden -> hidden
  nonlinearity = tanh

equation hadamard_merge : hidden -> hidden
  einsum = "i,i->i"
  semiring = real

fan dual : hidden -> hidden
  branches = [branch_relu, branch_tanh]
  merge = hadamard_merge
"""
        prog = parse_ua(text, numpy_backend())
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
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation relu : hidden -> hidden
  nonlinearity = relu

path layer : hidden -> hidden = linear >> relu
  residual = true
  semiring = real
"""
        spec = parse_ua_spec(text)
        ps = spec.specs[0]
        assert isinstance(ps, PathSpec)
        assert ps.residual is True
        assert ps.residual_semiring == 'real'

    def test_residual_path_numpy_oracle(self):
        """residual path: output = relu(x) + x"""
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation relu : hidden -> hidden
  nonlinearity = relu

path skip : hidden -> hidden = relu
  residual = true
  semiring = real
"""
        prog = parse_ua(text, numpy_backend())
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog('skip', x)
        expected = np.maximum(0.0, x) + x  # relu(x) + x
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_path_without_residual_unchanged(self):
        """Non-residual paths still work (backward compat)."""
        text = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation relu : hidden -> hidden
  nonlinearity = relu

path simple : hidden -> hidden = relu
"""
        spec = parse_ua_spec(text)
        ps = spec.specs[0]
        assert ps.residual is False


# ---------------------------------------------------------------------------
# Test: lens_fan declaration — parse, check spec, and error cases
# ---------------------------------------------------------------------------

_LENS_FAN_BASE = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)

equation fwd1 : hidden -> hidden
  nonlinearity = relu

equation bwd1 : hidden -> hidden
  nonlinearity = relu

equation fwd2 : hidden -> hidden
  nonlinearity = tanh

equation bwd2 : hidden -> hidden
  nonlinearity = tanh

equation merge_fwd : hidden -> hidden
  einsum = "i,i->i"
  semiring = real

equation merge_bwd : hidden -> hidden
  einsum = "i,i->i"
  semiring = real

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
lens_fan attention : hidden <-> hidden
  branches = [backprop1, backprop2]
  merge = merge_lens
"""
        spec = parse_ua_spec(text)
        assert len(spec.specs) == 1
        lfs = spec.specs[0]
        assert isinstance(lfs, LensFanSpec)
        assert lfs.name == 'attention'
        assert lfs.lens_names == ['backprop1', 'backprop2']
        assert lfs.merge_lens_name == 'merge_lens'
        assert lfs.domain_sort is not None
        assert lfs.codomain_sort is not None

    def test_lens_fan_unknown_lens_error(self):
        text = _LENS_FAN_BASE + """
lens_fan attention : hidden <-> hidden
  branches = [backprop1, nonexistent_lens]
  merge = merge_lens
"""
        with pytest.raises(ValueError, match="Unknown lens 'nonexistent_lens'"):
            parse_ua_spec(text)

    def test_lens_fan_unknown_merge_error(self):
        text = _LENS_FAN_BASE + """
lens_fan attention : hidden <-> hidden
  branches = [backprop1, backprop2]
  merge = nonexistent_merge
"""
        with pytest.raises(ValueError, match="Unknown merge lens 'nonexistent_merge'"):
            parse_ua_spec(text)


# ---------------------------------------------------------------------------
# Test: template equations (Para instantiation)
# ---------------------------------------------------------------------------

_TEMPLATE_BASE = """
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
sort hidden(real)
"""

_PROJ_TEMPLATE = _TEMPLATE_BASE + """
equation proj : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real
  template = true
"""


class TestTemplateEquations:

    def test_template_equation_not_in_equations(self):
        """An equation with template=true must NOT appear in spec.equations."""
        spec = parse_ua_spec(_PROJ_TEMPLATE)
        assert len(spec.equations) == 0

    def test_template_expansion_in_fan(self):
        """Fan with proj[q], proj[k], proj[v] should expand to 3 concrete equations."""
        text = _PROJ_TEMPLATE + """
equation merge : hidden -> hidden
  einsum = "i,i->i"
  semiring = real

fan kv : hidden -> hidden
  branches = [proj[q], proj[k], proj[v]]
  merge = merge
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [EquationView(eq).name for eq in spec.equations]
        assert 'q_proj' in eq_names_in_spec
        assert 'k_proj' in eq_names_in_spec
        assert 'v_proj' in eq_names_in_spec
        fan_spec = spec.specs[0]
        assert fan_spec.branch_names == ['q_proj', 'k_proj', 'v_proj']

    def test_template_expansion_in_path(self):
        """Path with proj[q] >> proj[k] should expand to two concrete equations."""
        text = _PROJ_TEMPLATE + """
path qk : hidden -> hidden = proj[q] >> proj[k]
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [EquationView(eq).name for eq in spec.equations]
        assert 'q_proj' in eq_names_in_spec
        assert 'k_proj' in eq_names_in_spec
        path_spec = spec.specs[0]
        assert path_spec.eq_names == ['q_proj', 'k_proj']

    def test_template_reuse_same_prefix(self):
        """Using proj[q] twice in the same path must produce exactly ONE q_proj equation."""
        text = _PROJ_TEMPLATE + """
path qq : hidden -> hidden = proj[q] >> proj[q]
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [EquationView(eq).name for eq in spec.equations]
        q_proj_count = eq_names_in_spec.count('q_proj')
        assert q_proj_count == 1

    def test_template_unknown_raises(self):
        """Referencing an undeclared template name should raise ValueError."""
        text = _TEMPLATE_BASE + """
path bad : hidden -> hidden = unknown[x]
"""
        with pytest.raises(ValueError, match="Unknown template 'unknown'"):
            parse_ua_spec(text)

    def test_template_mixed_refs(self):
        """Mix of plain idents and template refs in a path should both resolve."""
        text = _TEMPLATE_BASE + """
equation proj : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real
  template = true

equation relu : hidden -> hidden
  nonlinearity = relu

path qr : hidden -> hidden = proj[q] >> relu
"""
        spec = parse_ua_spec(text)
        eq_names_in_spec = [EquationView(eq).name for eq in spec.equations]
        assert 'q_proj' in eq_names_in_spec
        assert 'relu' in eq_names_in_spec
        path_spec = spec.specs[0]
        assert path_spec.eq_names == ['q_proj', 'relu']
