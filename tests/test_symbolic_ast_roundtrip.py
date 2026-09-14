import sympy as sp

from sciona.ghost.symbolic import serialize_expr, deserialize_expr


def test_roundtrip_preserves_uncancelled_expression_and_its_singularity():
    x = sp.Symbol("x")
    expression = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    encoded = serialize_expr(expression)
    restored = deserialize_expr(encoded)
    assert serialize_expr(restored) == encoded
    assert restored != sp.Integer(1)
    assert restored.subs(x, 0) is sp.nan


def test_roundtrip_preserves_unevaluated_equation_and_nested_arithmetic():
    x = sp.Symbol("x")
    lhs = sp.Add(x, sp.Integer(0), evaluate=False)
    equation = sp.Eq(lhs, x, evaluate=False)
    encoded = serialize_expr(equation)
    restored = deserialize_expr(encoded)
    assert isinstance(restored, sp.Equality)
    assert serialize_expr(restored) == encoded


def test_integral_roundtrip_does_not_inject_identity_factors():
    x = sp.Symbol("x")
    expression = sp.Integral(sp.cos(x), (x, 0, 1))
    encoded = serialize_expr(expression)
    for _ in range(3):
        expression = deserialize_expr(serialize_expr(expression))
        assert serialize_expr(expression) == encoded
    assert expression.doit() == sp.sin(1)


def test_quantum_state_roundtrip_preserves_bra_ket_order():
    from sympy.physics.quantum import Bra, Ket, InnerProduct

    expression = InnerProduct(Bra("a"), Ket("b"))
    encoded = serialize_expr(expression)
    assert serialize_expr(deserialize_expr(encoded)) == encoded
