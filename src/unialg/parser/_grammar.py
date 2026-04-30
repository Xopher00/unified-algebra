"""Grammar rules for the .ua DSL.

Builds Hydra parser combinators that transform .ua source text into
raw declaration tuples.  No semantic resolution — that lives in _resolver.py.
"""
from __future__ import annotations

from ._pratt import parse_pratt


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

    def _tok_literal(text, kind, value=None):
        return P.bind(P.string(text), lambda _: P.pure((kind, value)))

    def _tokenize(raw_token):
        return P.bind(_iws, lambda _:
               P.many(P.bind(raw_token, lambda t:
               P.bind(_iws, lambda _: P.pure(t)))))

    def _literal_token(spec):
        text, kind, *value = spec
        return _tok_literal(text, kind, value[0] if value else None)

    def _token_choice(*, literals=(), parsers=()):
        return P.choice(tuple(_literal_token(s) for s in literals) + tuple(parsers))

    def _pratt_expr(raw_token, *, label, binding_powers, nud, led, eof_token):
        return P.bind(_tokenize(raw_token), lambda toks:
               P.pure(parse_pratt(
                   list(toks),
                   label=label,
                   binding_powers=binding_powers,
                   nud=nud,
                   led=led,
                   eof_token=eof_token,
               )))

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

    # String literal with JSON-style escapes, but .ua-local whitespace rules.
    _dq = P.char(ord('"'))
    _escape_char = P.choice((
        P.bind(P.char(ord('"')), lambda _: P.pure(ord('"'))),
        P.bind(P.char(ord('\\')), lambda _: P.pure(ord('\\'))),
        P.bind(P.char(ord('/')), lambda _: P.pure(ord('/'))),
        P.bind(P.char(ord('b')), lambda _: P.pure(ord('\b'))),
        P.bind(P.char(ord('f')), lambda _: P.pure(ord('\f'))),
        P.bind(P.char(ord('n')), lambda _: P.pure(ord('\n'))),
        P.bind(P.char(ord('r')), lambda _: P.pure(ord('\r'))),
        P.bind(P.char(ord('t')), lambda _: P.pure(ord('\t'))),
    ))
    _string_char = P.alt(
        P.bind(P.char(ord('\\')), lambda _: _escape_char),
        P.satisfy(lambda c: c not in (ord('"'), ord('\\'), ord('\n'), ord('\r'))),
    )
    _raw_str = P.bind(_dq, lambda _:
               P.bind(P.many(_string_char), lambda cs:
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
    _unsigned_number = P.alt(_inf, _plain_num)
    _signed_number = P.bind(P.optional(P.char(ord('-'))), lambda sign:
                     P.bind(_unsigned_number, lambda v:
                     P.pure(-v if not isinstance(sign, Nothing) else v)))
    number_lit = _tok(_signed_number)

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

    _sort_sig = P.bind(_sort_or_product, lambda dom:
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

    _DT_NAME = 'dn'; _DT_NUM = 'd#'
    _DT_PLUS = 'd+'; _DT_MINUS = 'd-'; _DT_STAR = 'd*'; _DT_SLASH = 'd/'
    _DT_LP = 'd('; _DT_RP = 'd)'; _DT_COMMA = 'd,'

    _DEFINE_BP = {
        _DT_STAR: (70, 71), _DT_SLASH: (70, 71),
        _DT_PLUS: (60, 61), _DT_MINUS: (60, 61),
    }
    _DEFINE_OP = {
        _DT_PLUS: 'add', _DT_MINUS: 'subtract',
        _DT_STAR: 'multiply', _DT_SLASH: 'divide',
    }

    _define_tok_raw = _token_choice(
        literals=(
            ('+', _DT_PLUS), ('-', _DT_MINUS), ('*', _DT_STAR), ('/', _DT_SLASH),
            ('(', _DT_LP), (')', _DT_RP), (',', _DT_COMMA),
        ),
        parsers=(
            P.bind(_unsigned_number, lambda v: P.pure((_DT_NUM, v))),
            P.bind(_raw_ident, lambda name: P.pure((_DT_NAME, name))),
        ),
    )

    def _define_args(p):
        return p.parse_args(close=_DT_RP, sep=_DT_COMMA)

    def _define_nud(p, t):
        if t[0] == _DT_NUM: return ('lit', t[1])
        if t[0] == _DT_NAME:
            name = t[1]
            if p.peek()[0] == _DT_LP:
                p.advance()
                return ('call', name, _define_args(p))
            return ('var', name)
        if t[0] == _DT_MINUS:
            return ('call', 'neg', [p.parse(80)])
        if t[0] == _DT_LP:
            e = p.parse(0); p.expect(_DT_RP, 'unclosed ('); return e
        raise ValueError(f"define: unexpected token {t}")

    def _define_led(p, left, t, r_bp):
        return ('call', _DEFINE_OP[t[0]], [left, p.parse(r_bp)])

    _define_expr = _pratt_expr(
        _define_tok_raw,
        label="define",
        binding_powers=_DEFINE_BP,
        nud=_define_nud,
        led=_define_led,
        eof_token=(_DT_RP, None),
    )

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
                  P.bind(_define_expr, lambda body:
                  P.bind(_eol, lambda _:
                  P.pure(('define', name, ar, params, body)))))))))))

    # -------------------------------------------------------------------
    # Polynomial-expression Pratt parser — body of `functor` decls
    #
    # Operators:  &  product  (left-assoc, bp 70)
    #             +  sum      (left-assoc, bp 60)
    #             @  compose  (right-assoc, bp 80) — reserved, not yet supported
    # Atoms:      0  1  X  <sort-ident>  ( poly_expr )
    # -------------------------------------------------------------------

    _PT_ZERO = 'p0'; _PT_ONE = 'p1'; _PT_ID = 'pX'; _PT_NAME = 'pn'
    _PT_PLUS = 'p+'; _PT_AMP = 'p&'; _PT_AT = 'p@'
    _PT_LPAREN = 'p('; _PT_RPAREN = 'p)'

    _POLY_BP = {_PT_AT: (80, 79), _PT_AMP: (70, 71), _PT_PLUS: (60, 61)}
    _POLY_TAG = {_PT_PLUS: 'poly_sum', _PT_AMP: 'poly_prod', _PT_AT: 'poly_compose'}

    _poly_tok_raw = _token_choice(
        literals=(
            ('0', _PT_ZERO), ('1', _PT_ONE), ('+', _PT_PLUS), ('&', _PT_AMP),
            ('@', _PT_AT), ('(', _PT_LPAREN), (')', _PT_RPAREN),
            ('*', 'ERROR', "use '&' for functor product, not '*'"),
        ),
        parsers=(
            P.bind(_raw_ident, lambda n: P.pure((_PT_ID, None) if n == 'X' else (_PT_NAME, n))),
        ),
    )

    def _poly_nud(p, t):
        if t[0] == _PT_ZERO: return ('poly_zero',)
        if t[0] == _PT_ONE: return ('poly_one',)
        if t[0] == _PT_ID: return ('poly_id',)
        if t[0] == _PT_NAME: return ('poly_const', t[1])
        if t[0] == _PT_LPAREN:
            e = p.parse(0); p.expect(_PT_RPAREN, 'unclosed ('); return e
        if t[0] == 'ERROR': raise ValueError(f"functor: {t[1]}")
        raise ValueError(f"functor: unexpected {t[0]}")

    def _poly_led(p, left, t, r_bp):
        return (_POLY_TAG[t[0]], left, p.parse(r_bp))

    _poly_expr = _pratt_expr(
        _poly_tok_raw,
        label="functor",
        binding_powers=_POLY_BP,
        nud=_poly_nud,
        led=_poly_led,
        eof_token=(_PT_RPAREN, None),
    )

    # functor <name> : <poly_expr>
    #   [category = (set | poset)]
    functor_decl = P.bind(sym('functor'), lambda _:
                   P.bind(ident, lambda name:
                   P.bind(sym(':'), lambda _:
                   P.bind(_poly_expr, lambda body:
                   P.bind(_eol, lambda _:
                   P.bind(P.many(_indented_kv()), lambda attrs:
                   P.pure(('functor', name, body, dict(attrs)))))))))

    # -------------------------------------------------------------------
    # Cell-expression Pratt parser
    #
    # Operators (infix):
    #   &   parallel / product   (left-assoc, bp 70)
    #   >   sequence / flow      (left-assoc, bp 60)
    #   ~   lens / optic         (non-assoc, bp 50)
    #
    # Postfix (only after ~):
    #   *[R]  residual sort annotation (optional)
    #
    # Prefix atoms:
    #   ^[A]  copy     ![A]  delete     _[A]  identity
    #   >[F](...)  cata     <[F](...)  ana
    #   <number>   literal  (<expr>)   grouping
    #   <ident>    equation reference
    #
    # Named constructors (parsed as atoms):
    #   seq(f, g)  par(f, g)  lens(f, g) [*[R]]
    #   id[A]  copy[A]  drop[A]
    #   fold[F](...)  unfold[F](...)
    # -------------------------------------------------------------------

    _CT_NAME = 'n'; _CT_NUM = '#'
    _CT_GT = '>'; _CT_AMP = '&'; _CT_TILDE = '~'
    _CT_STAR_BR = '*['; _CT_CARET = '^['; _CT_BANG = '!['; _CT_UNDER = '_['
    _CT_GT_BR = '>['; _CT_LT_BR = '<['
    _CT_LP = '('; _CT_RP = ')'; _CT_COMMA = ','; _CT_LB = '['; _CT_RB = ']'

    _CELL_BP = {_CT_AMP: (70, 71), _CT_GT: (60, 61), _CT_TILDE: (50, 51)}
    _NAMED_BINARY = {'seq': 'cell_seq', 'par': 'cell_par'}
    _NAMED_BRACKET = {'id': 'cell_iden', 'copy': 'cell_copy', 'drop': 'cell_delete'}
    _NAMED_HOM = {'fold': 'cell_cata', 'unfold': 'cell_ana'}

    _cell_tok_raw = _token_choice(
        literals=(
            (_CT_STAR_BR, _CT_STAR_BR), (_CT_GT_BR, _CT_GT_BR), (_CT_LT_BR, _CT_LT_BR),
            (_CT_CARET, _CT_CARET), (_CT_BANG, _CT_BANG), (_CT_UNDER, _CT_UNDER),
            (_CT_GT, _CT_GT), (_CT_AMP, _CT_AMP), (_CT_TILDE, _CT_TILDE),
            (_CT_LP, _CT_LP), (_CT_RP, _CT_RP), (_CT_COMMA, _CT_COMMA),
            (_CT_LB, _CT_LB), (_CT_RB, _CT_RB),
            (';', 'ERROR', "use '>' not ';'"), ('*', 'ERROR', "use '&' not '*'"),
        ),
        parsers=(
            P.bind(_signed_number, lambda v: P.pure((_CT_NUM, v))),
            P.bind(_raw_ident, lambda name:
                   P.bind(P.many(P.satisfy(lambda c: chr(c) in "'?")), lambda mods:
                   P.pure((_CT_NAME, name + ''.join(chr(c) for c in mods))))),
        ),
    )

    def _cell_args(p):
        return p.parse_args(close=_CT_RP, sep=_CT_COMMA)

    def _bracket_name(p, label):
        n = p.expect(_CT_NAME, f"name inside {label}[]")
        p.expect(_CT_RB, f"closing ] for {label}[]")
        return n[1]

    def _hom(p, tag):
        f = p.expect(_CT_NAME, "functor name")
        p.expect(_CT_RB, "] after functor")
        p.expect(_CT_LP, "( after functor ref")
        return (tag, f[1], _cell_args(p))

    def _cell_nud(p, t):
        if t[0] == 'ERROR': raise ValueError(f"cell: {t[1]}")
        if t[0] == _CT_NAME:
            name = t[1]
            if name in _NAMED_BINARY and p.peek()[0] == _CT_LP:
                p.advance()
                args = _cell_args(p)
                if len(args) != 2: raise ValueError(f"{name}() takes 2 args")
                return (_NAMED_BINARY[name], args[0], args[1])
            if name == 'lens' and p.peek()[0] == _CT_LP:
                p.advance()
                args = _cell_args(p)
                if len(args) != 2: raise ValueError("lens() takes 2 args")
                residual = None
                if p.peek()[0] == _CT_STAR_BR:
                    p.advance()
                    residual = p.expect(_CT_NAME, "sort after *[")[1]
                    p.expect(_CT_RB, "] for *[")
                return ('cell_lens', args[0], args[1], residual)
            if name in (_NAMED_BRACKET | _NAMED_HOM) and p.peek()[0] == _CT_LB:
                p.advance()
                inner = _bracket_name(p, name)
                if name in _NAMED_BRACKET: return (_NAMED_BRACKET[name], inner)
                if name in _NAMED_HOM:
                    p.expect(_CT_LP, f"( after {name}[{inner}]")
                    return (_NAMED_HOM[name], inner, _cell_args(p))
            return ('cell_eq', name)
        if t[0] == _CT_NUM: return ('cell_lit', t[1])
        if t[0] == _CT_CARET: return ('cell_copy', _bracket_name(p, '^'))
        if t[0] == _CT_BANG: return ('cell_delete', _bracket_name(p, '!'))
        if t[0] == _CT_UNDER: return ('cell_iden', _bracket_name(p, '_'))
        if t[0] == _CT_GT_BR: return _hom(p, 'cell_cata')
        if t[0] == _CT_LT_BR: return _hom(p, 'cell_ana')
        if t[0] == _CT_LP:
            e = p.parse(0); p.expect(_CT_RP, 'unclosed ('); return e
        raise ValueError(f"cell: unexpected token {t}")

    def _cell_led(p, left, t, r_bp):
        right = p.parse(r_bp)
        if t[0] == _CT_TILDE:
            residual = None
            if p.peek()[0] == _CT_STAR_BR:
                p.advance()
                residual = p.expect(_CT_NAME, "sort name after *[")[1]
                p.expect(_CT_RB, "closing ] for *[")
            return ('cell_lens', left, right, residual)
        tag = 'cell_seq' if t[0] == _CT_GT else 'cell_par'
        return (tag, left, right)

    _cell_expr = _pratt_expr(
        _cell_tok_raw,
        label="cell",
        binding_powers=_CELL_BP,
        nud=_cell_nud,
        led=_cell_led,
        eof_token=(_CT_RP, None),
    )

    # cell <name> : <sort_sig> = <cell_expr>
    cell_decl = P.bind(sym('cell'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym(':'), lambda _:
                P.bind(_sort_sig, lambda sig:
                P.bind(sym('='), lambda _:
                P.bind(_cell_expr, lambda expr:
                P.bind(_eol, lambda _:
                P.pure(('cell', name, sig, expr)))))))))

    decl = P.choice((
        define_decl,
        import_decl, algebra_decl, spec_decl, op_decl,
        share_decl, functor_decl, cell_decl,
    ))

    # -------------------------------------------------------------------
    # Full program: blank lines + declarations
    # -------------------------------------------------------------------
    program = P.bind(skip_blank, lambda _:
              P.bind(P.many(P.bind(decl, lambda d:
                            P.bind(skip_blank, lambda _:
                            P.pure(d)))), lambda decls:
              P.pure(list(decls))))

    return program
