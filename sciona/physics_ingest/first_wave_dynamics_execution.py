"""Symbolic and sampled constant-mass dynamics from an explicit position tree.

One inertial-frame Cartesian component, SI units. The caller supplies a twice
 differentiable position on the evaluation domain. This differentiates the
position; it does not numerically integrate forces or certify global smoothness.
No evaluation of source strings, Python code, or stored legacy srepr.
"""
import math
import sympy as sp

TIME = sp.Symbol('t', real=True)
FUNCTIONS = {'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'exp':sp.exp,'log':sp.log,
             'sinh':sp.sinh,'cosh':sp.cosh,'atan':sp.atan}


def position_expression(tree):
    count=0
    def parse(node,depth=0):
        nonlocal count
        count+=1
        if count>1024 or depth>32: raise ValueError('Expression exceeds structural limits')
        if node=='t': return TIME
        if type(node) in (int,float):
            if not math.isfinite(node): raise ValueError('Finite coefficients required')
            return sp.Rational(str(node))
        if not isinstance(node,dict) or set(node)!={'op','args'} or not isinstance(node['args'],list):
            raise ValueError('Invalid position expression tree')
        op=node['op']; args=node['args']
        if op=='position' and not args: return sp.Function('x')(TIME)
        if op in ('add','mul') and len(args)>=2:
            return (sp.Add if op=='add' else sp.Mul)(*(parse(a,depth+1) for a in args))
        if op=='pow' and len(args)==2:
            return sp.Pow(parse(args[0],depth+1),parse(args[1],depth+1))
        if isinstance(op,str) and op in FUNCTIONS and len(args)==1:
            return FUNCTIONS[op](parse(args[0],depth+1))
        raise ValueError('Unsupported expression operator/arity')
    return parse(tree)


def execute_dynamics(payload):
    if not isinstance(payload,dict) or set(payload)!={'version','mass','position','times'} or type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('Version1 dynamics payload required')
    mass=payload['mass']
    if type(mass) not in (int,float) or not math.isfinite(mass) or mass<=0:
        raise ValueError('Positive finite constant mass required')
    times=payload['times']
    if not isinstance(times,list) or any(type(t) not in (int,float) or not math.isfinite(t) for t in times):
        raise ValueError('Finite sample times required')
    position=position_expression(payload['position'])
    acceleration=sp.diff(position,TIME,2)
    force=sp.Rational(str(mass))*acceleration
    solved=sp.cancel(force/sp.Rational(str(mass)))
    if sp.simplify(solved-acceleration)!=0: raise ArithmeticError('Acceleration solve identity failed')
    values=[]
    for time in times:
        row=[]
        for expr in (position,acceleration,force):
            number=expr.subs(TIME,sp.Rational(str(time))).evalf(17)
            if number.is_real is not True or number.is_finite is not True:
                raise ValueError('Trajectory is not finite and evaluable at requested time')
            value=float(number)
            if not math.isfinite(value): raise ValueError('Numerical output overflow')
            row.append(value)
        values.append(row)
    return dict(version=1,position=sp.srepr(position),acceleration=sp.srepr(acceleration),
        force=sp.srepr(force),acceleration_from_force=sp.srepr(solved),
        times=list(times),samples=values,
        assumptions=['Constant positive mass','One inertial Cartesian component',
                     'Position twice differentiable on evaluation domain; coefficients in consistent SI units'],
        sample_columns=['position','acceleration','force'])
