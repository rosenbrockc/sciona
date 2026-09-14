"""Corrected standard x-axis Lorentz boost and invariant-interval certificate."""
import sympy as sp

SOURCE_VERSION='e2b3b291-b9f3-50bf-be6c-fe2d390f27b7'
SOURCE_HASH='c03df95d9bff24d13e35624a4157a790cadd266c8679d4e6e25f919071aa52b4'


def build_proof():
    b,g=sp.symbols('beta gamma',real=True)
    c=sp.Symbol('c',positive=True)
    S=sp.Symbol('gamma_squared',positive=True)
    v=b*c
    return dict(gamma_squared=1/(1-b*b),positive_gamma=1/sp.sqrt(1-b*b),
                boost=sp.Matrix([[g,-g*b,0,0],[-g*b,g,0,0],[0,0,1,0],[0,0,0,1]]),
                source_x_coefficient=S-c*c*(1-S)**2/(S*v*v),
                source_xt_coefficient=-2*S*v-2*c*c*(1-S)/v,
                source_time_coefficient=S*(c*c-v*v),
                source_time_correction=(1-S)/(S*v))


def verify_proof(proof):
    if proof!=build_proof():raise ValueError('Reviewed Lorentz reconstruction changed')
    b,g=sp.symbols('beta gamma',real=True)
    c=sp.Symbol('c',positive=True)
    S=sp.Symbol('gamma_squared',positive=True)
    q=proof['gamma_squared'];B=proof['boost'];eta=sp.diag(1,-1,-1,-1)
    reduce=lambda e:sp.cancel(sp.expand(e).subs(g**2,q))
    metric=(B.T*eta*B-eta).applyfunc(reduce)
    inverse=(B.xreplace({b:-b})*B-sp.eye(4)).applyfunc(reduce)
    checks={
        'metric_preserved':metric==sp.zeros(4),
        'velocity_reversal_inverse':inverse==sp.zeros(4),
        'proper_determinant':reduce(B.det()-1)==0,
        'zero_velocity_identity':B.subs({b:0,g:1})==sp.eye(4),
        'positive_branch_square':sp.cancel(proof['positive_gamma']**2-q)==0,
        'positive_branch_at_zero':proof['positive_gamma'].subs(b,0)==1,
        'all_null_x_coefficient':sp.cancel(proof['source_x_coefficient'].subs(S,q)-1)==0,
        'all_null_cross_coefficient':sp.cancel(proof['source_xt_coefficient'].subs(S,q))==0,
        'all_null_time_coefficient':sp.cancel(proof['source_time_coefficient'].subs(S,q)-c*c)==0,
        'time_transform_reconstruction':sp.cancel(proof['source_time_correction'].subs(S,q)+b/c)==0,
    }
    if not all(checks.values()):raise ValueError('Lorentz invariant or coefficient failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_steps_reviewed=23,source_ast_parity=False,checks=checks,
                assumptions=['Linear reciprocal standard boost along x with common origin and unchanged y,z.',
                             'Positive light speed c and signed relative velocity |v|<c; beta=v/c.',
                             'Light-cone preservation for all events, not a single null observation.',
                             'Positive gamma chosen by continuity from identity and future time orientation.'],
                quotient_domains='Source v and 1-gamma² divisions apply only to nonzero v; zero-velocity identity established separately.',
                corrected_transform='On (ct,x,y,z), ct_prime=gamma*(ct-beta*x), x_prime=gamma*(x-beta*ct), transverse coordinates unchanged.',
                limitations=['Original malformed expansion, denominator, symbol aliases and missing AST terms require reconstruction.',
                             'No acceleration, general relativity, arbitrary boost direction or superluminal frame.',
                             'The x² coefficient alone has an extraneous gamma²=1 branch; remaining coefficients must also hold.',
                             'Invariant proof does not promise exact cancellation after independent float64 rounding.'])
