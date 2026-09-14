"""Normalized finite-interval Dirichlet modes with explicit spectral domains."""
import sympy as sp

SOURCE_VERSION = '56d398db-3d6c-5144-a061-3b8c19ab598d'
SOURCE_HASH = '2502a41c3f5c0a036b1838f3ba0d522dfba6ece9fef3e6f2919648a58ce89832'


def symbols():
    x, phase = sp.symbols('x phase', real=True)
    width = sp.Symbol('width', positive=True)
    n = sp.Symbol('n', integer=True, positive=True)
    return x, width, n, phase


def build_proof():
    x, width, n, phase = symbols()
    k = n*sp.pi/width
    amplitude = sp.sqrt(2/width)*sp.exp(sp.I*phase)
    wave = amplitude*sp.sin(k*x)
    return dict(wave=wave, opposite_wave=-wave, wavenumber=k, eigenvalue=k*k,
                derivative=amplitude*k*sp.cos(k*x), second_derivative=-k*k*wave)


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed reconstruction changed')
    x, width, n, phase = symbols()
    wave, k = proof['wave'], proof['wavenumber']
    density = sp.simplify(wave*sp.conjugate(wave))
    primitive = x/2-sp.sin(2*k*x)/(4*k)
    checks = dict(first_derivative=sp.diff(wave, x)-proof['derivative'],
        second_derivative=sp.diff(wave, x, 2)-proof['second_derivative'],
        eigen_equation=-sp.diff(wave, x, 2)-proof['eigenvalue']*wave,
        left_boundary=wave.subs(x, 0), right_boundary=wave.subs(x, width),
        normalized_density=density-2*sp.sin(k*x)**2/width,
        normalization=sp.integrate(density, (x, 0, width))-1,
        sine_square_antiderivative=sp.diff(primitive, x)-sp.sin(k*x)**2,
        sine_square_definite_integral=primitive.subs(x, width)-primitive.subs(x, 0)-width/2,
        positive_representative=wave.subs(phase, 0)-sp.sqrt(2/width)*sp.sin(k*x),
        negative_representative=wave.subs(phase, sp.pi)-proof['opposite_wave'].subs(phase, 0),
        opposite_eigen_equation=sp.diff(proof['opposite_wave'], x, 2)+k*k*proof['opposite_wave'],
        opposite_density=sp.simplify(proof['opposite_wave']*sp.conjugate(proof['opposite_wave']))-density,
        derivative_energy=sp.integrate(sp.simplify(proof['derivative']*sp.conjugate(proof['derivative'])), (x, 0, width))-k*k)
    if any(sp.simplify(v) != 0 for v in checks.values()):
        raise ValueError('Dirichlet mode identity failed')
    # For eigenvalue zero the general solution is linear, not a*sin(0*x)+b.
    a, b = sp.symbols('a b')
    zero_solution = a*x+b
    zero_coefficients = sp.solve([zero_solution.subs(x, 0), zero_solution.subs(x, width)], [a, b])
    if zero_coefficients != {a: 0, b: 0}:
        raise ValueError('Zero-eigenvalue branch not excluded')
    # A negative eigenvalue has exponential basis; boundary determinant never
    # vanishes for positive real decay rate and width.
    decay = sp.Symbol('decay', positive=True)
    determinant = sp.exp(-decay*width)-sp.exp(decay*width)
    negative_form = -2*sp.sinh(decay*width)
    if sp.simplify(determinant-negative_form.rewrite(sp.exp)) != 0 or negative_form.is_negative is not True:
        raise ValueError('Negative-eigenvalue boundary determinant not nonzero')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=31, source_ast_parity=False,
                reconstructed_checks={k: True for k in checks},
                zero_eigenvalue_excluded=True, negative_eigenvalue_excluded=True,
                assumptions=['Finite positive interval width; integer n>=1; real x in [0,width].',
                             'Twice differentiable interior eigenfunction of -d²/dx² with zero endpoint values.',
                             'Unit L2 norm on the interval; constant finite real phase.',
                             'Positive mode indices label independent eigenspaces; negative indices only change a global sign.'],
                limitations=['ODE and derivatives are interior formulas; endpoint values impose Dirichlet conditions.',
                             'No derivative matching or distributional extension outside the interval.',
                             'Normalization fixes amplitude magnitude; real plus/minus signs are special global phases.',
                             'No finite potential well, quantum mass/energy conversion, or arbitrary boundary condition claim.',
                             'This establishes individual normalized modes, not completeness of arbitrary expansions.'])
