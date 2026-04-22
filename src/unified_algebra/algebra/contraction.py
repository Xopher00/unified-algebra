"""Tensor contraction over a user-defined semiring.

Patterned after torch_semiring_einsum (bdusell): the equation is compiled
once into index-alignment structures, then the contraction is executed by
plugging in three callbacks resolved from the semiring:

    multiply_in_place(a, b)  — elementwise ⊗
    sum_block(a, dims)       — reduction ⊕ over dims
    add_in_place(a, b)       — elementwise ⊕ for accumulating blocks

This module is backend-agnostic: all tensor operations go through the
backend abstraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backend import Backend
    from .semiring import ResolvedSemiring


# ---------------------------------------------------------------------------
# Compiled equation
# ---------------------------------------------------------------------------

class Equation:
    """A pre-compiled einsum equation."""

    def __init__(self, input_vars, output_vars, num_vars, var_locations):
        self.input_vars = input_vars      # list of lists of int
        self.output_vars = output_vars    # list of int
        self.num_vars = num_vars
        self.var_locations = var_locations # var_id -> [(arg_idx, dim_idx)]
        self._reduced_vars = None
        self._reduced_dims = None

    @property
    def reduced_vars(self):
        if self._reduced_vars is None:
            out_set = set(self.output_vars)
            seen = set()
            self._reduced_vars = []
            for var_list in self.input_vars:
                for v in var_list:
                    if v not in out_set and v not in seen:
                        self._reduced_vars.append(v)
                        seen.add(v)
        return self._reduced_vars

    @property
    def reduced_dims(self):
        """Dimension indices of reduced vars in the aligned tensor
        (output_vars ++ reduced_vars)."""
        if self._reduced_dims is None:
            n_out = len(self.output_vars)
            self._reduced_dims = tuple(range(n_out, n_out + len(self.reduced_vars)))
        return self._reduced_dims

    def get_sizes(self, args, variables):
        return [args[loc[0][0]].shape[loc[0][1]]
                for v in variables
                for loc in [self.var_locations[v]]]

    def validate(self, args):
        for locs in self.var_locations:
            if len(locs) > 1:
                size = args[locs[0][0]].shape[locs[0][1]]
                for arg_i, dim_i in locs[1:]:
                    if args[arg_i].shape[dim_i] != size:
                        raise ValueError(
                            f"Dimension mismatch at arg {arg_i} dim {dim_i}")


def compile_equation(equation: str) -> Equation:
    """Compile an einsum equation string into an Equation object."""
    lhs, rhs = equation.split("->")
    arg_strs = lhs.split(",")
    char_to_int = {}
    var_locations = []
    input_vars = []
    for arg_i, arg_str in enumerate(arg_strs):
        arg_vars = []
        for dim_i, ch in enumerate(arg_str):
            var_id = char_to_int.get(ch)
            if var_id is None:
                var_id = len(char_to_int)
                char_to_int[ch] = var_id
                var_locations.append([])
            var_locations[var_id].append((arg_i, dim_i))
            arg_vars.append(var_id)
        input_vars.append(arg_vars)
    output_vars = [char_to_int[ch] for ch in rhs]
    return Equation(input_vars, output_vars, len(char_to_int), var_locations)


# ---------------------------------------------------------------------------
# Contraction execution
# ---------------------------------------------------------------------------

def semiring_contract(
    equation: Equation,
    args,
    sr: ResolvedSemiring,
    backend: Backend,
    block_size: int | None = None,
):
    """Execute a semiring contraction.

    Args:
        equation: compiled equation
        args: input tensors
        sr: resolved semiring (callbacks already resolved)
        backend: provides structural tensor ops (expand_dims, transpose)
        block_size: when set, the first reduced variable is sliced into chunks
            of at most this many elements, and partial results are accumulated
            with ⊕_elementwise.  None (the default) uses the existing single-
            pass code path.  The result is numerically identical to the
            unblocked version because ⊕ is associative.
    """
    equation.validate(args)

    reduced_vars = equation.reduced_vars

    # Fast path: no blocking requested, or no reduction dimensions at all.
    if block_size is None or not reduced_vars:
        return _contract_full(equation, args, sr, backend)

    # Determine the size of the first reduced variable so we can decide
    # whether blocking is actually needed.
    first_reduced_var = reduced_vars[0]
    first_reduced_size = args[
        equation.var_locations[first_reduced_var][0][0]
    ].shape[equation.var_locations[first_reduced_var][0][1]]

    if first_reduced_size <= block_size:
        # All reduction fits in one block — no splitting needed.
        return _contract_full(equation, args, sr, backend)

    # Blocked path: slice the first reduced variable into chunks and
    # accumulate partial contractions with ⊕_elementwise.
    accumulator = None
    for start in range(0, first_reduced_size, block_size):
        end = min(start + block_size, first_reduced_size)
        sliced_args = _slice_args(args, equation, first_reduced_var, start, end)
        partial = _contract_full(equation, sliced_args, sr, backend)
        if accumulator is None:
            accumulator = partial
        else:
            accumulator = sr.plus_elementwise(accumulator, partial)

    return accumulator


def _contract_full(equation: Equation, args, sr: ResolvedSemiring, backend: Backend):
    """Single-pass contraction — the original algorithm."""
    # Target dim order: output_vars ++ reduced_vars
    all_target_vars = equation.output_vars + equation.reduced_vars

    # Align each tensor to target dims
    factors = []
    for arg, arg_vars in zip(args, equation.input_vars):
        factors.append(_align_tensor(arg, arg_vars, all_target_vars, backend))

    # Elementwise ⊗
    term = factors[0]
    for f in factors[1:]:
        term = sr.times_elementwise(term, f)

    # Reduce contracted dims with ⊕
    return sr.plus_reduce(term, equation.reduced_dims)


def _slice_args(args, equation: Equation, reduced_var: int, start: int, end: int):
    """Return a new args list where every occurrence of *reduced_var* is sliced
    to the index range [start:end] along its corresponding dimension.

    All other dimensions of each argument tensor are left unchanged.
    """
    sliced = []
    for arg, arg_vars in zip(args, equation.input_vars):
        if reduced_var in arg_vars:
            # Build a slice tuple: ':' for every dim except the reduced one.
            dim = arg_vars.index(reduced_var)
            idx = tuple(
                slice(start, end) if d == dim else slice(None)
                for d in range(len(arg_vars))
            )
            sliced.append(arg[idx])
        else:
            sliced.append(arg)
    return sliced


def _align_tensor(tensor, tensor_vars, target_vars, backend):
    """Unsqueeze and permute a tensor so its dims align with target_vars."""
    # Build index: which position does each var occupy in the tensor?
    var_to_dim = {v: i for i, v in enumerate(tensor_vars)}

    ndim = len(tensor_vars)
    extra = 0
    perm = []
    result = tensor
    for tv in target_vars:
        if tv in var_to_dim:
            perm.append(var_to_dim[tv])
        else:
            # This var isn't in this tensor — unsqueeze a size-1 dim
            result = backend.expand_dims(result, axis=ndim + extra)
            perm.append(ndim + extra)
            extra += 1

    return backend.transpose(result, perm)


