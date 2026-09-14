"""Signed one-dimensional constant-acceleration equations with quotient domains."""
import sympy as sp

SOURCE_VERSION='41c11625-ad96-5738-82bc-aa6fb7f95a10'
SOURCE_HASH='2c8a5c5ef6fd2b7bc5c989dcaabb25e199c19e1895acbe29c4ced1d09df1d27b'


def build_proof():
    a,t,v0,v,d=sp.symbols('a t v0 v d',real=True)
    E=lambda lhs,rhs:sp.Eq(lhs,rhs,evaluate=False)
    half=sp.Rational(1,2)
    steps=[E(a*t,v-v0),E(a*t+v0,v),E(v,v0+a*t),
           E(d/t,(v+v0)/2),E(d,(v+v0)*t/2),
           E(d,((v0+a*t)+v0)*t/2),E(d,(2*v0+a*t)*t/2),
           E(d,v0*t+half*a*t**2),E(v**2,(v0+a*t)**2),
           E(v**2,v0**2+2*a*(v0*t+half*a*t**2)),
           E(v**2,v0**2+2*a*d),E(v-v0,a*t),E(a*t,v-v0),
           E(t,(v-v0)/a),E(d,(v+v0)*(v-v0)/(2*a)),
           E(d,(v**2-v0**2)/(2*a)),E(2*a*d,v**2-v0**2),
           E(2*a*d+v0**2,v**2),E(v**2,v0**2+2*a*d),
           E(v-a*t,v0),E(d,(v-a*t)*t+half*a*t**2),
           E(d,v*t-a*t**2+half*a*t**2),E(d,v*t-half*a*t**2)]
    return dict(velocity=v0+a*t,displacement=v0*t+half*a*t**2,steps=steps)


def verify_proof(proof):
    if proof!=build_proof():raise ValueError('Reviewed reconstruction changed')
    a,t,v0,v,d=sp.symbols('a t v0 v d',real=True)
    V,D=proof['velocity'],proof['displacement']
    for i,e in enumerate(proof['steps'],1):
        if sp.cancel((e.lhs-e.rhs).subs({v:V,d:D},simultaneous=True))!=0:
            raise ValueError('Motion identity failed at step '+str(i))
    s=sp.Symbol('s',real=True)
    checks={'velocity_integral':v0+sp.integrate(a,(s,0,t))-V,
            'displacement_integral':sp.integrate(v0+a*s,(s,0,t))-D,
            'initial_velocity':V.subs(t,0)-v0,
            'initial_displacement':D.subs(t,0),
            'velocity_derivative':sp.diff(D,t)-V,
            'constant_acceleration':sp.diff(V,t)-a,
            'zero_acceleration_velocity':V.subs(a,0)-v0,
            'zero_acceleration_displacement':D.subs(a,0)-v0*t}
    if any(sp.expand(r)!=0 for r in checks.values()):raise ValueError('Independent integral check failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_steps_reviewed=23,reconstructed_steps_validated=23,source_ast_parity=False,
                integration_checks={key:True for key in checks},
                assumptions=['Signed one-dimensional velocity, signed displacement and constant acceleration.',
                             'Finite real initial velocity and acceleration; elapsed time t>=0.',
                             'Source quotient definitions and step4 require t!=0; steps14to16 require a!=0.'],
                endpoint_extensions='Polynomial integration proves t=0 and a=0 cases independently of quotient branches.',
                limitations=['d is displacement, not path length when motion reverses.',
                             'Endpoint average velocity equals time average only under the constant-acceleration assumption.',
                             'Squared velocity equation is a consequence; inverse use alone cannot select signed velocity.',
                             'No variable-acceleration or multidimensional path-length result.'])
