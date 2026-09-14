"""Hyperbolic identities from exponential definitions, including complex arguments."""
import sympy as sp

SOURCE_VERSION = 'a6bee7b6-17cb-5f68-abe7-f56912e79dc2'
SOURCE_HASH = '3b0f152719793a679aade985da120125efa5ac123df9a110600efbb43a9b7d81'


def build_proof():
    x = sp.Symbol('x', real=True)
    p, q = sp.exp(x), sp.exp(-x)
    s, c = (p-q)/2, (p+q)/2
    S, C, T, H = sp.sinh(x), sp.cosh(x), sp.tanh(x), sp.sech(x)
    eq = lambda lhs, rhs: sp.Eq(lhs, rhs, evaluate=False)
    return [eq(S**2, s**2), eq(C**2, c**2),
            eq(C**2-S**2, c**2-s**2),
            eq(C**2-S**2, (sp.exp(2*x)+2+sp.exp(-2*x)-(sp.exp(2*x)-2+sp.exp(-2*x)))/4),
            eq(C**2-S**2, 1),
            eq(sp.sin(sp.I*x), (q-p)/(2*sp.I)),
            eq(sp.sin(sp.I*x), sp.I*S),
            eq(sp.I*S, (q-p)/(2*sp.I)),
            eq(sp.cos(sp.I*x), (q+p)/2),
            eq(sp.cos(sp.I*x), C),
            eq(H, 2/(p+q)), eq(T, s/C), eq(T, (p-q)/(p+q)),
            eq(T**2, (p-q)**2/(p+q)**2),
            eq(H**2, 4/(p+q)**2),
            eq(H**2+T**2, (4+(p-q)**2)/(p+q)**2),
            eq(H**2+T**2, (sp.exp(2*x)+2+sp.exp(-2*x))/(p+q)**2),
            eq(H**2+T**2, 1)]


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed hyperbolic reconstruction changed')
    x = sp.Symbol('x', real=True)
    order = [1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for number in order:
        equation = proof[number-1]
        if sp.simplify((equation.lhs-equation.rhs).rewrite(sp.exp)) != 0:
            raise ValueError('Exponential-definition residual failed at step '+str(number))
    # A positive exponential and its reciprocal establish a nonzero denominator
    # for every real x. This is false for general complex inputs.
    if (sp.exp(x)+sp.exp(-x)).is_positive is not True:
        raise ValueError('Quotient domain not established')
    z = sp.Symbol('z', complex=True, finite=True)
    sine = (sp.exp(sp.I*z)-sp.exp(-sp.I*z))/(2*sp.I)
    cosine = (sp.exp(sp.I*z)+sp.exp(-sp.I*z))/2
    checks = [sp.diff(sine, z)-cosine, sp.diff(cosine, z)+sine,
              sine.subs(z, 0), cosine.subs(z, 0)-1,
              sine.subs(z, sp.I*x)-sp.I*sp.sinh(x).rewrite(sp.exp),
              cosine.subs(z, sp.I*x)-sp.cosh(x).rewrite(sp.exp)]
    if any(sp.simplify(v) != 0 for v in checks):
        raise ValueError('Complex exponential definitions failed')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=18, source_ast_parity=False,
                reconstructed_steps_validated=18, replay_order=order,
                complex_definition_checks=6, denominator_positive=True,
                assumptions=['Finite real dimensionless x; I is the mathematical imaginary unit.',
                             'Entire complex sine/cosine and hyperbolic exponential definitions.'],
                prerequisite_scope='Complex exponential definitions are checked directly; no substitution into a real-only Euler certificate.',
                limitations=['No literal uncorrected source AST replay.',
                             'Tanh and sech quotients are not globally defined for complex x; real domain is deliberate.',
                             'Algebraic certificates do not promise stable evaluation by subtracting rounded cosh/sinh squares.'])
