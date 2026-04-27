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

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unialg.backend import Backend
    from unialg.algebra.semiring import Semiring

CONTRACTION_REGISTRY: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Compiled equation
# ---------------------------------------------------------------------------

_BATCH_CHAR_POOL = "bcdefghmnopqrstuvwxyz"


class CompiledEinsum:
    """A pre-compiled einsum equation."""

    def __init__(self, input_vars, output_vars, num_vars, var_locations, char_to_int=None):
        self.input_vars = input_vars      # list of lists of int
        self.output_vars = output_vars    # list of int
        self.num_vars = num_vars
        self.var_locations = var_locations # var_id -> [(arg_idx, dim_idx)]
        self._char_to_int = char_to_int   # original char→var_id mapping, for to_string()

    def to_string(self) -> str:
        """Render back to einsum form. Requires char_to_int (preserved by compile_einsum)."""
        if self._char_to_int is None:
            raise ValueError("CompiledEinsum has no source char map; cannot render to string")
        int_to_char = {v: c for c, v in self._char_to_int.items()}
        lhs = ",".join("".join(int_to_char[v] for v in vars) for vars in self.input_vars)
        rhs = "".join(int_to_char[v] for v in self.output_vars)
        return f"{lhs}->{rhs}"

    def prepend_batch_var(self) -> "CompiledEinsum":
        """Return a new CompiledEinsum with a fresh batch var prepended to every operand.

        Picks the first unused char from _BATCH_CHAR_POOL ("bcdefgh...") so the
        common case of einsum strings using "ijk..." gets 'b' as the batch dim.
        """
        used = set(self._char_to_int or {})
        batch_char = next(c for c in _BATCH_CHAR_POOL if c not in used)
        new_var_id = self.num_vars
        new_char_to_int = {**(self._char_to_int or {}), batch_char: new_var_id}
        # Existing vars: dim_i += 1 since batch occupies dim 0
        new_var_locations = [
            [(arg_i, dim_i + 1) for arg_i, dim_i in locs] for locs in self.var_locations
        ]
        # Batch var: dim 0 of every input
        new_var_locations.append([(arg_i, 0) for arg_i in range(len(self.input_vars))])
        return CompiledEinsum(
            input_vars=[[new_var_id] + vars for vars in self.input_vars],
            output_vars=[new_var_id] + self.output_vars,
            num_vars=self.num_vars + 1,
            var_locations=new_var_locations,
            char_to_int=new_char_to_int,
        )

    @cached_property
    def reduced_vars(self):
        out_set = set(self.output_vars)
        seen = set()
        result = []
        for var_list in self.input_vars:
            for v in var_list:
                if v not in out_set and v not in seen:
                    result.append(v)
                    seen.add(v)
        return result

    @cached_property
    def reduced_dims(self):
        n_out = len(self.output_vars)
        return tuple(range(n_out, n_out + len(self.reduced_vars)))

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


def compile_einsum(einsum: str) -> CompiledEinsum:
    """Compile an einsum string into a CompiledEinsum object."""
    lhs, rhs = einsum.split("->")
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
    return CompiledEinsum(input_vars, output_vars, len(char_to_int), var_locations, char_to_int)


# ---------------------------------------------------------------------------
# Contraction execution
# ---------------------------------------------------------------------------

def _auto_block_size(einsum: CompiledEinsum, args, backend: Backend, budget_fraction=0.5):
    """Compute a block_size that keeps the intermediate tensor under memory budget."""
    avail = backend.available_memory()
    if avail is None:
        return None

    budget = int(avail * budget_fraction)
    elem_bytes = getattr(args[0], 'itemsize', 8)

    all_target_vars = einsum.output_vars + einsum.reduced_vars
    target_sizes = einsum.get_sizes(args, all_target_vars)
    intermediate_elems = 1
    for s in target_sizes:
        intermediate_elems *= s
    intermediate_bytes = intermediate_elems * elem_bytes

    if intermediate_bytes <= budget:
        return None

    output_sizes = einsum.get_sizes(args, einsum.output_vars)
    output_elems = 1
    for s in output_sizes:
        output_elems *= s

    other_reduced_elems = 1
    for v in einsum.reduced_vars[1:]:
        other_reduced_elems *= einsum.get_sizes(args, [v])[0]

    per_slice_elems = output_elems * other_reduced_elems
    per_slice_bytes = per_slice_elems * elem_bytes
    if per_slice_bytes == 0:
        return None

    return max(1, budget // per_slice_bytes)


def semiring_contract(
    einsum: CompiledEinsum,
    args,
    sr: Semiring.Resolved,
    backend: Backend,
    block_size: int | None = None,
):
    """Execute a semiring contraction.

    Args:
        einsum: compiled einsum
        args: input tensors
        sr: resolved semiring (callbacks already resolved)
        backend: provides structural tensor ops (expand_dims, transpose)
        block_size: explicit override for block size. When None (default),
            the block size is computed automatically based on available memory.
            The result is numerically identical to the unblocked version
            because the semiring's plus is associative.
    """
    einsum.validate(args)

    reduced_vars = einsum.reduced_vars
    hook = sr.contraction_fn

    def _full(einsum_, args_):
        if hook is not None:
            factors = _align_factors(einsum_, args_, backend)
            cs = _make_compute_sum(factors, einsum_.reduced_dims)
            return hook(cs, backend)
        return _contract_full(einsum_, args_, sr, backend)

    if not reduced_vars:
        return _full(einsum, args)

    if hook is not None:
        return _full(einsum, args)

    if block_size is None:
        block_size = _auto_block_size(einsum, args, backend)

    if block_size is None:
        return _full(einsum, args)

    first_reduced_var = reduced_vars[0]
    first_reduced_size = args[
        einsum.var_locations[first_reduced_var][0][0]
    ].shape[einsum.var_locations[first_reduced_var][0][1]]

    if first_reduced_size <= block_size:
        return _full(einsum, args)

    accumulator = None
    for start in range(0, first_reduced_size, block_size):
        end = min(start + block_size, first_reduced_size)
        sliced_args = _slice_args(args, einsum, first_reduced_var, start, end)
        partial = _full(einsum, sliced_args)
        if accumulator is None:
            accumulator = partial
        else:
            accumulator = sr.plus_elementwise(accumulator, partial)

    return accumulator


def _align_factors(einsum: CompiledEinsum, args, backend: Backend):
    """Align all input tensors to the target dim order: output_vars ++ reduced_vars."""
    all_target_vars = einsum.output_vars + einsum.reduced_vars
    return [_align_tensor(arg, arg_vars, all_target_vars, backend)
            for arg, arg_vars in zip(args, einsum.input_vars)]


def _make_compute_sum(factors, reduced_dims):
    """Build a compute_sum(times_fn, reduce_fn) callable from pre-aligned factors.

    This is the engine's primitive for multi-pass contraction. A contraction_fn
    hook can call it multiple times with different operations.

    Attributes on the returned callable:
        compute_product(times_fn) — returns the elementwise product before reduction
        reduced_dims — tuple of axis indices to reduce over
    """
    def compute_product(times_fn):
        term = factors[0]
        for f in factors[1:]:
            term = times_fn(term, f)
        return term

    def compute_sum(times_fn, reduce_fn):
        return reduce_fn(compute_product(times_fn), reduced_dims)

    compute_sum.compute_product = compute_product
    compute_sum.reduced_dims = reduced_dims
    return compute_sum


def _contract_full(einsum: CompiledEinsum, args, sr: Semiring.Resolved, backend: Backend):
    """Single-pass contraction — the default algorithm."""
    factors = _align_factors(einsum, args, backend)
    cs = _make_compute_sum(factors, einsum.reduced_dims)
    return cs(sr.times_elementwise, sr.plus_reduce)


def _slice_args(args, einsum: CompiledEinsum, reduced_var: int, start: int, end: int):
    """Return a new args list where every occurrence of *reduced_var* is sliced
    to the index range [start:end] along its corresponding dimension.

    All other dimensions of each argument tensor are left unchanged.
    """
    sliced = []
    for arg, arg_vars in zip(args, einsum.input_vars):
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


# ---------------------------------------------------------------------------
# High-level contraction + nonlinearity
# ---------------------------------------------------------------------------

def contract_and_apply(compiled, tensor_args, sr, backend, nl_fn=None, params=()):
    """Contract tensors via compiled einsum, then apply optional nonlinearity."""
    r = semiring_contract(compiled, tensor_args, sr, backend) if compiled else tensor_args[0]
    return nl_fn(r, *params) if nl_fn else r


def contract_merge(compiled, tensors, sr, backend, nl_fn=None, n_inputs=2, name=""):
    """Reduce a list of tensors via contraction. N-ary when arity matches, binary fold for arity 2."""
    if n_inputs == 1:
        if len(tensors) != 1:
            raise ValueError(f"Unary merge '{name}' expects 1-element list, got {len(tensors)}")
        result = semiring_contract(compiled, [tensors[0]], sr, backend)
    elif n_inputs == len(tensors):
        result = semiring_contract(compiled, tensors, sr, backend)
    elif n_inputs == 2:
        result = tensors[0]
        for t in tensors[1:]:
            result = semiring_contract(compiled, [result, t], sr, backend)
    else:
        raise ValueError(
            f"Merge '{name}' has {n_inputs}-operand einsum but received "
            f"{len(tensors)} tensors — binary fold only works for 2-operand einsums")
    return nl_fn(result) if nl_fn else result
