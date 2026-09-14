"""Corrected variance identity under a normalized linear expectation."""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION = '90bfde0a-dfb2-5514-98b4-9ed0944dbc41'
SOURCE_HASH = '3dd3b23ab004f016bd071b06a518a274131a27d9d122ed7e73b839c3a9620872'


def symbols():
    x, mean, second = sp.symbols('x mean second', real=True)
    return x, mean, second, sp.Function('Expectation')


def normalized_expectation(expression):
    x, mean, second, _ = symbols()
    if expression.free_symbols - {x, mean, second}:
        raise ValueError('Unreviewed expectation symbol')
    polynomial = sp.Poly(sp.expand(expression), x)
    if polynomial.degree() > 2:
        raise ValueError('Only finite first and second moments are assumed')
    return sp.expand(sum(polynomial.nth(k)*moment for k, moment in enumerate([1, mean, second])))


@dataclass(frozen=True)
class VarianceProof:
    expressions: tuple


def build_proof():
    x, mean, second, expectation = symbols()
    third = sp.Add(second, -2*sp.Mul(mean, mean, evaluate=False), mean**2, evaluate=False)
    fourth = sp.Add(second, -2*mean**2, mean**2, evaluate=False)
    return VarianceProof((expectation((x-mean)**2), expectation(x*x-2*x*mean+mean*mean),
                          third, fourth, second-mean**2))


def verify_proof(proof):
    expected = build_proof(); x, mean, second, expectation = symbols()
    if len(proof.expressions) != 5 or proof.expressions[0] != expected.expressions[0] or proof.expressions[-1] != expected.expressions[-1]:
        raise ValueError('Variance definition and terminal identity required')
    for index, expression in enumerate(proof.expressions):
        if expression.free_symbols - {x, mean, second}:
            raise ValueError('Unreviewed proof symbol')
        if index < 2:
            if expression.func != expectation or len(expression.args) != 1:
                raise ValueError('Expectation of expanded real square required')
        elif expression.has(expectation) or expression.has(x):
            raise ValueError('Only scalar moments may remain after expectation')
    first, expanded, linear, combined, final = proof.expressions
    if sp.expand(first.args[0]-expanded.args[0]) != 0:
        raise ValueError('Incorrect square expansion')
    if sp.expand(normalized_expectation(expanded.args[0])-linear) != 0:
        raise ValueError('Normalized expectation or mean substitution differs')
    if sp.expand(linear-combined) != 0 or sp.expand(combined-final) != 0:
        raise ValueError('Mean square cannot become the second moment')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH, source_ast_parity=False,
                expressions=[serialize_expr(e) for e in proof.expressions],
                assumptions=['Real random variable with finite second moment.', 'Normalized linear expectation: E[1]=1 and mean=E[x].'],
                checks=dict(square_expansion=True, normalized_linearity=True, mean_square_preserved=True, final_identity=True),
                correction='Source intermediate -2 E[x^2] replaced with -2 E[x]^2.',
                numerical_realization='Finite nonnegative weighted distribution, normalized by positive total weight; population variance, not unbiased sample variance.')
