"""Explicit correction of the generated first-wave Newtonian derivation.

The legacy derivative parse is not reused. This proof supplies the missing
kinematic premise and nonzero constant-mass condition. No original approval claim.
"""
import sympy as sp


def corrected_proof():
    t=sp.Symbol('t',real=True)
    m=sp.Symbol('m',positive=True)
    force=sp.Function('F')(t)
    acceleration=sp.Function('a')(t)
    position=sp.Function('x')(t)
    newton=sp.Eq(force,m*acceleration,evaluate=False)
    solved=sp.Eq(acceleration,force/m,evaluate=False)
    kinematics=sp.Eq(acceleration,sp.diff(position,t,2),evaluate=False)
    result=sp.Eq(force,m*sp.diff(position,t,2),evaluate=False)
    solve_residual=sp.simplify((newton.lhs-newton.rhs).subs(acceleration,solved.rhs))
    substitution_residual=sp.simplify((newton.lhs-newton.rhs).subs(acceleration,kinematics.rhs)-(result.lhs-result.rhs))
    return dict(equations={k:sp.srepr(v) for k,v in dict(newton=newton,solved=solved,kinematics=kinematics,result=result).items()},
        checks=dict(solve_residual_zero=solve_residual==0,substitution_residual_zero=substitution_residual==0,
                    actual_second_derivative=result.rhs.has(sp.Derivative(position,(t,2)))),
        assumptions=['Positive constant mass','One Cartesian component in an inertial frame',
                     'Twice differentiable position','Acceleration equals second time derivative of position'],
        dimensions={'t':'T','m':'M','x':'L','a':'L/T^2','F':'M*L/T^2'},
        limitations=['Original source graph omits kinematic premise and nonzero mass condition',
                     'Legacy symbolic derivative parse is invalid','Not a time integration algorithm'])
