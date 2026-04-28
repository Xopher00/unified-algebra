"""Grammar rules for the .ua DSL.

Builds Hydra parser combinators that transform .ua source text into
raw declaration tuples.  No semantic resolution — that lives in _resolver.py.
"""
from __future__ import annotations

from functools import reduce


def _build_parser():
    """Build the full .ua program parser.  Returns the top-level Parser[list[tuple]]."""
    import hydra.parsers as P
    from hydra.parsers import Nothing

    # -------------------------------------------------------------------
    # Primitives
    # -------------------------------------------------------------------

    # Whitespace — inline only (no newlines, which are meaningful)
    _iws = P.many(P.satisfy(lambda c: chr(c) in ' \t'))

    def _tok(p):
        """Wrap p with inline-whitespace on both sides."""
        return P.bind(_iws, lambda _:
               P.bind(p, lambda x:
               P.bind(_iws, lambda __:
               P.pure(x))))

    def sym(s):
        return _tok(P.string(s))

    _newline = P.satisfy(lambda c: chr(c) in '\n\r')
    _not_nl = P.satisfy(lambda c: chr(c) not in '\n\r')

    # End-of-line: inline ws + optional comment + newline
    _eol = P.bind(_iws, lambda _:
           P.bind(P.optional(P.bind(P.char(ord('#')), lambda _:
                             P.bind(P.many(_not_nl), lambda _:
                             P.pure(None)))), lambda _:
           P.bind(_newline, lambda _:
           P.pure(None))))

    _hash_comment_nl = P.bind(P.char(ord('#')), lambda _:
                       P.bind(P.many(_not_nl), lambda _:
                       P.bind(_newline, lambda _: P.pure(None))))
    _ws_line = P.bind(P.some(P.satisfy(lambda c: chr(c) in ' \t')), lambda _:
               P.bind(P.alt(_hash_comment_nl, P.bind(_newline, lambda _: P.pure(None))),
               lambda x: P.pure(x)))
    _blank_line = P.alt(P.bind(_newline, lambda _: P.pure(None)),
                  P.alt(_hash_comment_nl, _ws_line))
    skip_blank = P.many(_blank_line)

    # Identifier
    _id_start = P.satisfy(lambda c: chr(c).isalpha() or chr(c) == '_')
    _id_rest = P.many(P.satisfy(lambda c: chr(c).isalnum() or chr(c) == '_'))
    _raw_ident = P.bind(_id_start, lambda c:
                 P.bind(_id_rest, lambda cs:
                 P.pure(chr(c) + ''.join(chr(x) for x in cs))))
    ident = _tok(_raw_ident)

    # String literal  "…"
    _dq = P.char(ord('"'))
    _not_dq = P.satisfy(lambda c: c != ord('"'))
    _raw_str = P.bind(_dq, lambda _:
               P.bind(P.many(_not_dq), lambda cs:
               P.bind(_dq, lambda _:
               P.pure(''.join(chr(c) for c in cs)))))
    string_lit = _tok(_raw_str)

    # Number: optional minus, digits, optional decimal, or 'inf'/'-inf'
    _digit = P.satisfy(lambda c: chr(c).isdigit())
    _digits = P.some(_digit)
    _decimal = P.bind(P.char(ord('.')), lambda _:
               P.bind(_digits, lambda ds:
               P.pure('.' + ''.join(chr(c) for c in ds))))
    _plain_num = P.bind(_digits, lambda ds:
                 P.bind(P.optional(_decimal), lambda mf:
                 P.pure(float(
                     ''.join(chr(c) for c in ds) +
                     (mf.value if not isinstance(mf, Nothing) else '')
                 ))))
    _inf = P.bind(P.string('inf'), lambda _: P.pure(float('inf')))
    _neg_inf = P.bind(P.string('-inf'), lambda _: P.pure(float('-inf')))
    number_lit = _tok(P.alt(_neg_inf, P.alt(_inf, _plain_num)))

    # Attr parser must fail without consuming when no indent (P.some guarantees this)
    _indent = P.bind(P.some(P.satisfy(lambda c: chr(c) in ' \t')), lambda _: P.pure(None))
    _comma = sym(',')

    # Boolean literals
    _true = P.bind(P.string('true'), lambda _: P.pure(True))
    _false = P.bind(P.string('false'), lambda _: P.pure(False))
    _bool_lit = _tok(P.alt(_false, _true))

    _any_value = P.alt(string_lit, P.alt(_bool_lit, P.alt(number_lit, ident)))

    # Comma-separated ident list — used for list-valued attributes (e.g. inputs)
    _ident_csv = P.bind(ident, lambda first:
                 P.bind(P.many(P.bind(_comma, lambda _: ident)), lambda rest:
                 P.pure([first] + list(rest))))

    def _indented_kv():
        return P.bind(_indent, lambda _:
               P.bind(ident, lambda k:
               P.bind(sym('='), lambda _:
               P.bind(_any_value, lambda v:
               P.bind(_eol, lambda _:
               P.pure((k, v)))))))

    # Like _indented_kv but parses list-valued attributes for specified keys.
    # When the key matches a list key, the value is parsed as a comma-separated
    # ident list. All other keys fall back to _any_value.
    _LIST_ATTR_KEYS = {'inputs'}

    def _indented_kv_op():
        def _value_parser(k):
            return _ident_csv if k in _LIST_ATTR_KEYS else _any_value
        return P.bind(_indent, lambda _:
               P.bind(ident, lambda k:
               P.bind(sym('='), lambda _:
               P.bind(_value_parser(k), lambda v:
               P.bind(_eol, lambda _:
               P.pure((k, v)))))))

    # -------------------------------------------------------------------
    # Sort type signatures — factored over the separator
    # -------------------------------------------------------------------

    _product_sort = P.bind(sym('('), lambda _:
                    P.bind(ident, lambda first:
                    P.bind(P.some(P.bind(_comma, lambda _: ident)), lambda rest:
                    P.bind(sym(')'), lambda _:
                    P.pure(('_product', tuple([first] + list(rest))))))))

    _sort_or_product = P.alt(_product_sort, ident)

    _sort_sig = P.bind(ident, lambda dom:
                P.bind(sym('->'), lambda _:
                P.bind(_sort_or_product, lambda cod:
                P.pure((dom, cod)))))

    # -------------------------------------------------------------------
    # Individual declaration parsers
    # -------------------------------------------------------------------

    def _kv(k, vp):
        return P.bind(sym(k), lambda _:
               P.bind(sym('='), lambda _: vp))

    def _trailing_sr_kv():
        """Parse a single optional trailing ', key=ident' pair in algebra(...)."""
        return P.bind(_comma, lambda _:
               P.bind(ident, lambda k:
               P.bind(sym('='), lambda _:
               P.bind(ident, lambda v:
               P.pure((k, v))))))

    _sr_args = P.bind(sym('('), lambda _:
               P.bind(_kv('plus', ident), lambda plus:
               P.bind(_comma, lambda _:
               P.bind(_kv('times', ident), lambda times:
               P.bind(_comma, lambda _:
               P.bind(_kv('zero', number_lit), lambda zero:
               P.bind(_comma, lambda _:
               P.bind(_kv('one', number_lit), lambda one:
               P.bind(P.many(_trailing_sr_kv()), lambda trailing:
               P.bind(sym(')'), lambda _:
               P.pure({**dict(trailing), **dict(plus=plus, times=times, zero=zero, one=one,
                              contraction=dict(trailing).get('contraction', ''),
                              residual=dict(trailing).get('residual', ''),
                              leq=dict(trailing).get('leq', ''))})))))))))))

    import_decl = P.bind(sym('import'), lambda _:
                  P.bind(ident, lambda name:
                  P.bind(_eol, lambda _:
                  P.pure(('import', name)))))

    algebra_decl = P.bind(sym('algebra'), lambda _:
                   P.bind(ident, lambda name:
                   P.bind(_sr_args, lambda kw_args:
                   P.bind(_eol, lambda _:
                   P.pure(('algebra', name, kw_args))))))

    _axis_size = P.bind(P.char(ord(':')), lambda _:
                 P.bind(_tok(_digits), lambda ds:
                 P.pure(int(''.join(chr(c) for c in ds)))))
    _axis_item = P.bind(ident, lambda name:
                 P.bind(P.optional(_axis_size), lambda mb_sz:
                 P.pure(name if isinstance(mb_sz, Nothing) else f"{name}:{mb_sz.value}")))

    _ident_list = P.bind(sym('['), lambda _:
                  P.bind(_axis_item, lambda first:
                  P.bind(P.many(P.bind(_comma, lambda _: _axis_item)), lambda rest:
                  P.bind(sym(']'), lambda _:
                  P.pure(tuple([first] + list(rest)))))))

    _spec_extra = P.optional(
        P.bind(_comma, lambda _:
        P.alt(
            P.bind(sym('batched'), lambda _:
            P.bind(P.optional(
                P.bind(_comma, lambda _:
                P.bind(_kv('axes', _ident_list), lambda a: P.pure(a)))
            ), lambda mb_axes:
            P.pure((True, mb_axes.value if not isinstance(mb_axes, Nothing) else ())))),
            P.bind(_kv('axes', _ident_list), lambda axes:
            P.pure((False, axes)))
        )))

    spec_decl = P.bind(sym('spec'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym('('), lambda _:
                P.bind(ident, lambda sr_name:
                P.bind(_spec_extra, lambda mb:
                P.bind(sym(')'), lambda _:
                P.bind(_eol, lambda _:
                P.pure(('spec', name, sr_name,
                        mb.value[0] if not isinstance(mb, Nothing) else False,
                        mb.value[1] if not isinstance(mb, Nothing) else ())))))))))

    op_decl = P.bind(sym('op'), lambda _:
              P.bind(ident, lambda name:
              P.bind(sym(':'), lambda _:
              P.bind(_sort_sig, lambda sig:
              P.bind(_eol, lambda _:
              P.bind(P.many(_indented_kv_op()), lambda kv_list:
              P.pure(('op', name, sig, dict(kv_list)))))))))

    def _sep_by(sep_str, p):
        _sep = sym(sep_str)
        return P.bind(p, lambda first:
               P.bind(P.many(P.bind(_sep, lambda _: p)), lambda rest:
               P.pure([first] + list(rest))))

    # share name : op1, op2, ..., opN — 2-cell weight-tying declaration.
    share_decl = P.bind(sym('share'), lambda _:
                 P.bind(ident, lambda name:
                 P.bind(sym(':'), lambda _:
                 P.bind(_sep_by(',', ident), lambda op_names:
                 P.bind(_eol, lambda _:
                 P.pure(('share', name, op_names)))))))


    # -------------------------------------------------------------------
    # Expression sub-grammar for `define` declarations
    # -------------------------------------------------------------------

    _INFIX_MAP = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}

    def _fold_infix(first, rest):
        result = first
        for op, rhs in rest:
            result = ('call', _INFIX_MAP[op], [result, rhs])
        return result

    _expr_cell = [None]
    _unary_cell = [None]
    _expr_fwd = P.bind(P.pure(None), lambda _: _expr_cell[0])
    _unary_fwd = P.bind(P.pure(None), lambda _: _unary_cell[0])

    _lit_expr = P.bind(number_lit, lambda v: P.pure(('lit', v)))
    _paren_expr = P.bind(P.char(ord('(')), lambda _:
                  P.bind(_iws, lambda _:
                  P.bind(_expr_fwd, lambda e:
                  P.bind(_iws, lambda _:
                  P.bind(P.char(ord(')')), lambda _:
                  P.bind(_iws, lambda _:
                  P.pure(e)))))))

    _ident_expr = P.bind(ident, lambda name:
                  P.alt(
                      P.bind(P.char(ord('(')), lambda _:
                      P.bind(_iws, lambda _:
                      P.bind(_expr_fwd, lambda first:
                      P.bind(P.many(P.bind(_comma, lambda _: _expr_fwd)), lambda rest:
                      P.bind(_iws, lambda _:
                      P.bind(P.char(ord(')')), lambda _:
                      P.bind(_iws, lambda _:
                      P.pure(('call', name, [first] + list(rest)))))))))),
                      P.pure(('var', name))
                  ))

    _atom_expr = P.alt(_lit_expr, P.alt(_paren_expr, _ident_expr))

    _neg_expr = P.bind(sym('-'), lambda _:
                P.bind(_unary_fwd, lambda e:
                P.pure(('call', 'neg', [e]))))
    _unary_expr = P.alt(_atom_expr, _neg_expr)
    _unary_cell[0] = _unary_expr

    _mul_op = P.alt(P.bind(sym('*'), lambda _: P.pure('*')),
                    P.bind(sym('/'), lambda _: P.pure('/')))
    _mul_div_expr = P.bind(_unary_expr, lambda first:
                    P.bind(P.many(P.bind(_mul_op, lambda op:
                                  P.bind(_unary_expr, lambda rhs:
                                  P.pure((op, rhs))))), lambda rest:
                    P.pure(_fold_infix(first, rest))))

    _add_op = P.alt(P.bind(sym('+'), lambda _: P.pure('+')),
                    P.bind(sym('-'), lambda _: P.pure('-')))
    _add_sub_expr = P.bind(_mul_div_expr, lambda first:
                    P.bind(P.many(P.bind(_add_op, lambda op:
                                  P.bind(_mul_div_expr, lambda rhs:
                                  P.pure((op, rhs))))), lambda rest:
                    P.pure(_fold_infix(first, rest))))

    _expr_cell[0] = _add_sub_expr

    # -------------------------------------------------------------------
    # define declaration
    # -------------------------------------------------------------------

    _arity_kw = P.alt(
        P.bind(sym('unary'), lambda _: P.pure('unary')),
        P.bind(sym('binary'), lambda _: P.pure('binary')))

    _param_list = P.bind(ident, lambda first:
                  P.bind(P.many(P.bind(_comma, lambda _: ident)), lambda rest:
                  P.pure([first] + list(rest))))

    define_decl = P.bind(sym('define'), lambda _:
                  P.bind(_arity_kw, lambda ar:
                  P.bind(ident, lambda name:
                  P.bind(sym('('), lambda _:
                  P.bind(_param_list, lambda params:
                  P.bind(sym(')'), lambda _:
                  P.bind(sym('='), lambda _:
                  P.bind(_add_sub_expr, lambda body:
                  P.bind(_eol, lambda _:
                  P.pure(('define', name, ar, params, body)))))))))))

    def _fold_left_binary(first, rest, tag):
        result = first
        for r in rest:
            result = (tag, result, r)
        return result

    # -------------------------------------------------------------------
    # Polynomial-expression sub-grammar — body of `functor` decls
    #
    # Atoms:    0  1  X  <sort-ident>  ( poly_expr )
    # Product:  atom * atom * ...      (level 3, left-assoc)
    # Sum:      prod + prod + ...      (level 4, left-assoc)
    # -------------------------------------------------------------------

    _poly_expr_cell = [None]
    _poly_expr_fwd = P.bind(P.pure(None), lambda _: _poly_expr_cell[0])

    _poly_zero = P.bind(_tok(P.string('0')), lambda _: P.pure(('poly_zero',)))
    _poly_one  = P.bind(_tok(P.string('1')), lambda _: P.pure(('poly_one',)))
    # `X` is a reserved sort identifier meaning the recursion variable.
    _poly_id   = P.bind(_tok(P.string('X')), lambda _: P.pure(('poly_id',)))
    _poly_const = P.bind(ident, lambda name:
                  P.pure(('poly_id',) if name == 'X' else ('poly_const', name)))
    _poly_paren = P.bind(sym('('), lambda _:
                  P.bind(_poly_expr_fwd, lambda e:
                  P.bind(sym(')'), lambda _: P.pure(e))))

    # Order matters: 0/1/X must be tried before generic ident, so the keywords
    # don't get consumed as sort names.
    _poly_atom = reduce(P.alt, [
        _poly_zero, _poly_one, _poly_id, _poly_paren, _poly_const,
    ])

    _poly_prod = P.bind(_poly_atom, lambda first:
                 P.bind(P.many(P.bind(sym('*'), lambda _:
                                P.bind(_poly_atom, lambda r: P.pure(r)))), lambda rest:
                 P.pure(_fold_left_binary(first, rest, 'poly_prod'))))

    _poly_sum = P.bind(_poly_prod, lambda first:
                P.bind(P.many(P.bind(sym('+'), lambda _:
                               P.bind(_poly_prod, lambda r: P.pure(r)))), lambda rest:
                P.pure(_fold_left_binary(first, rest, 'poly_sum'))))

    _poly_expr_cell[0] = _poly_sum

    # functor <name> : <poly_expr>
    #   [category = (set | poset)]
    functor_decl = P.bind(sym('functor'), lambda _:
                   P.bind(ident, lambda name:
                   P.bind(sym(':'), lambda _:
                   P.bind(_poly_sum, lambda body:
                   P.bind(_eol, lambda _:
                   P.bind(P.many(_indented_kv()), lambda attrs:
                   P.pure(('functor', name, body, dict(attrs)))))))))

    # -------------------------------------------------------------------
    # Cell-expression sub-grammar (operator-based DSL)
    #
    # Atoms:    <ident>  (...)  ^[A]  ![A]  _[A]  number  >[F](...)  <[F](...)
    # Tensor:   atom * atom * ...                       (level 3, left-assoc)
    # Seq:      tensor ; tensor ; ...                   (level 5, left-assoc)
    # Lens:     seq <-> seq             (height-1, level 6)
    #           seq <-> seq {R}         (height-2, residual sort R)
    # -------------------------------------------------------------------

    _cell_expr_cell = [None]
    _cell_expr_fwd = P.bind(P.pure(None), lambda _: _cell_expr_cell[0])

    # ^[A] / ![A] / _[A] — bracket-applied prefix on a sort identifier
    def _bracketed_ident(prefix):
        return P.bind(_tok(P.string(prefix + '[')), lambda _:
               P.bind(ident, lambda s:
               P.bind(sym(']'), lambda _:
               P.pure(s))))

    _cell_copy   = P.bind(_bracketed_ident('^'), lambda s: P.pure(('cell_copy',   s)))
    _cell_delete = P.bind(_bracketed_ident('!'), lambda s: P.pure(('cell_delete', s)))
    _cell_iden   = P.bind(_bracketed_ident('_'), lambda s: P.pure(('cell_iden',   s)))

    # >[F](c, c, ...) / <[F](c, c, ...) — bracket holds the functor, parens the cells
    def _hom_atom(prefix, tag):
        return P.bind(_tok(P.string(prefix + '[')), lambda _:
               P.bind(ident, lambda f_name:
               P.bind(sym(']'), lambda _:
               P.bind(sym('('), lambda _:
               P.bind(_cell_expr_fwd, lambda first:
               P.bind(P.many(P.bind(_comma, lambda _: _cell_expr_fwd)), lambda rest:
               P.bind(sym(')'), lambda _:
               P.pure((tag, f_name, [first] + list(rest))))))))))

    _cell_cata = _hom_atom('>', 'cell_cata')
    _cell_ana  = _hom_atom('<', 'cell_ana')

    # Numeric literal — auto-wraps as lit
    _cell_numlit = P.bind(number_lit, lambda v: P.pure(('cell_lit', v)))

    # Parens grouping
    _cell_paren = P.bind(sym('('), lambda _:
                  P.bind(_cell_expr_fwd, lambda e:
                  P.bind(sym(')'), lambda _: P.pure(e))))

    # Bare identifier — equation reference
    _cell_eq_ref = P.bind(ident, lambda name: P.pure(('cell_eq', name)))

    # Atom: try bracket-applied operators first, then number, parens, ident.
    _cell_atom = reduce(P.alt, [
        _cell_copy, _cell_delete, _cell_iden,
        _cell_cata, _cell_ana,
        _cell_numlit, _cell_paren, _cell_eq_ref,
    ])

    # Tensor product (level 3): atom * atom * ...
    _cell_tensor = P.bind(_cell_atom, lambda first:
                   P.bind(P.many(P.bind(sym('*'), lambda _:
                                  P.bind(_cell_atom, lambda r: P.pure(r)))), lambda rest:
                   P.pure(_fold_left_binary(first, rest, 'cell_par'))))

    # Sequential composition (level 5): tensor ; tensor ; ...
    _cell_seq = P.bind(_cell_tensor, lambda first:
                P.bind(P.many(P.bind(sym(';'), lambda _:
                               P.bind(_cell_tensor, lambda r: P.pure(r)))), lambda rest:
                P.pure(_fold_left_binary(first, rest, 'cell_seq'))))

    # Lens (level 6): seq <-> seq, with optional residual sort {R}
    _cell_residual_opt = P.optional(P.bind(sym('{'), lambda _:
                                    P.bind(ident, lambda r:
                                    P.bind(sym('}'), lambda _:
                                    P.pure(r)))))
    _cell_lens_tail = P.optional(P.bind(sym('<->'), lambda _:
                                  P.bind(_cell_seq, lambda rhs:
                                  P.bind(_cell_residual_opt, lambda res:
                                  P.pure((rhs, res))))))

    def _maybe_lens(first, tail):
        if isinstance(tail, Nothing):
            return first
        rhs, residual = tail.value
        residual_name = None if isinstance(residual, Nothing) else residual.value
        return ('cell_lens', first, rhs, residual_name)

    _cell_expr = P.bind(_cell_seq, lambda first:
                 P.bind(_cell_lens_tail, lambda tail:
                 P.pure(_maybe_lens(first, tail))))

    _cell_expr_cell[0] = _cell_expr

    # cell <name> : <sort_sig> = <cell_expr>
    cell_decl = P.bind(sym('cell'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym(':'), lambda _:
                P.bind(_sort_sig, lambda sig:
                P.bind(sym('='), lambda _:
                P.bind(_cell_expr, lambda expr:
                P.bind(_eol, lambda _:
                P.pure(('cell', name, sig, expr)))))))))

    decl = reduce(P.alt, [
        define_decl,
        import_decl, algebra_decl, spec_decl, op_decl,
        share_decl, functor_decl, cell_decl,
    ])

    # -------------------------------------------------------------------
    # Full program: blank lines + declarations
    # -------------------------------------------------------------------
    program = P.bind(skip_blank, lambda _:
              P.bind(P.many(P.bind(decl, lambda d:
                            P.bind(skip_blank, lambda _:
                            P.pure(d)))), lambda decls:
              P.pure(list(decls))))

    return program
