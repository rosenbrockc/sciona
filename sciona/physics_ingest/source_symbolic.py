"""Conservative inspection of upstream symbolic forms, without source approval.

Source payloads are untrusted inputs. Interpret a small expression grammar rather
than evaluating Python. An upstream expression is review evidence, not permission
to replace the original formula or its normalized representation.
"""
from __future__ import annotations

import ast
from typing import Any

from sciona.ghost.symbolic import deserialize_expr, serialize_expr


class SourceSymbolicError(ValueError):
    """A source form cannot be interpreted by the supported grammar."""


def parse_source_srepr(source: str) -> Any:
    import sympy as sp
    from sympy.physics.quantum import Bra, Ket, Dagger, Operator

    if not isinstance(source, str) or not source.strip() or len(source) > 100_000:
        raise SourceSymbolicError("missing_or_oversized_input")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise SourceSymbolicError("invalid_syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 4_000:
        raise SourceSymbolicError("expression_too_large")
    constructors = {name: getattr(sp, name) for name in (
        "Symbol", "Integer", "Rational", "Float", "Add", "Mul", "Pow",
        "Equality", "Tuple", "Derivative", "Abs", "conjugate", "exp",
        "sin", "cos", "tan", "sinh", "cosh", "tanh", "sech", "log",
        "LeviCivita",
    )}
    constructors.update(Bra=Bra, Ket=Ket, Dagger=Dagger, Operator=Operator)
    constants = {"E": sp.E, "I": sp.I, "pi": sp.pi, "oo": sp.oo}

    def visit(node: ast.AST, depth: int = 0):
        if depth > 100:
            raise SourceSymbolicError("expression_too_deep")
        descend = lambda child: visit(child, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = descend(node.operand)
            if type(value) not in (int, float):
                raise SourceSymbolicError("unsupported_negation")
            return -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return sp.Mul(descend(node.left), descend(node.right), evaluate=False)
        if isinstance(node, ast.Call):
            args = [descend(arg) for arg in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg not in {"precision", "real", "positive", "negative", "integer", "commutative"} or kw.arg in kwargs:
                    raise SourceSymbolicError("unsupported_keyword")
                kwargs[kw.arg] = descend(kw.value)
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name == "Function" and len(args) == 1 and type(args[0]) is str and not kwargs:
                    return sp.Function(args[0])
                if name == "Integral":
                    with sp.evaluate(True):
                        return sp.Integral(*args, **kwargs)
                if name in constructors:
                    return constructors[name](*args, **kwargs)
            # Undefined mathematical functions use Function('f')(x), never
            # arbitrary callable expressions or attribute access.
            if (isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name)
                    and node.func.func.id == "Function" and not kwargs):
                return descend(node.func)(*args)
        raise SourceSymbolicError("unsupported_syntax")

    try:
        with sp.evaluate(False):
            result = visit(tree.body)
        if not isinstance(result, sp.Basic):
            raise SourceSymbolicError("not_symbolic_expression")
        return result
    except SourceSymbolicError:
        raise
    except Exception as exc:
        raise SourceSymbolicError("constructor_failed") from exc


def inspect_source_symbolic(source_payload: dict, stored_srepr: str = "") -> dict:
    """Return private review evidence; exact AST agreement is not semantic proof."""
    import sympy as sp

    evidence = {
        "runner_version": "source-symbolic-inspection.v1",
        "scope": "upstream syntax and AST correspondence only; no source or semantic approval",
        "status": "unavailable",
        "correspondence": "unchecked",
    }
    raw = source_payload.get("raw_payload") or {}
    if not isinstance(raw, dict):
        return {**evidence, "reason": "unsupported_payload"}
    if raw.get("latex_relation") != "=":
        return {**evidence, "reason": "unsupported_relation"}
    try:
        lhs = parse_source_srepr(raw.get("sympy_lhs"))
        rhs_text = raw.get("sympy_rhs")
        if isinstance(rhs_text, str) and rhs_text.strip():
            rhs = parse_source_srepr(rhs_text)
            if isinstance(lhs, sp.Equality) or isinstance(rhs, sp.Equality):
                raise SourceSymbolicError("nested_relation")
            expression = sp.Eq(lhs, rhs, evaluate=False)
        elif isinstance(lhs, sp.Equality):
            expression = lhs
        else:
            raise SourceSymbolicError("missing_rhs")
        serialized = serialize_expr(expression)
        if serialize_expr(deserialize_expr(serialized)) != serialized:
            raise SourceSymbolicError("roundtrip_failed")
        evidence.update(status="roundtrip_passed", sympy_srepr=serialized)
        evidence["correspondence"] = (
            "exact_ast_match" if stored_srepr == serialized else
            "different_ast_requires_review" if stored_srepr else "no_stored_expression"
        )
    except SourceSymbolicError as exc:
        evidence.update(status="unsupported", reason=str(exc))
    except Exception:
        evidence.update(status="unsupported", reason="roundtrip_failed")
    return evidence
