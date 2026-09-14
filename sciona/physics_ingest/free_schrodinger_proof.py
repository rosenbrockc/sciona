"""Free Schrödinger plane-wave consistency and finite-superposition proof."""
import sympy as sp

SOURCE_VERSION = '3e30bcec-9752-5d2f-90f9-c5e806d421d9'
SOURCE_HASH = '72c2732025dd2f9b0873efb81fb5c5ee6f9fa8e2d898c64a2558693e64746619'


def symbols():
    x, y, z, t = sp.symbols('x y z t', real=True)
    px, py, pz = sp.symbols('px py pz', real=True)
    m, hbar = sp.symbols('m hbar', positive=True)
    ar, ai = sp.symbols('ar ai', real=True)
    return (x, y, z), t, (px, py, pz), m, hbar, ar+sp.I*ai


def laplacian(field, coordinates):
    return sum(sp.diff(field, coordinate, 2) for coordinate in coordinates)


def residual(field, coordinates, time, mass, hbar):
    return sp.I*hbar*sp.diff(field, time)+hbar**2*laplacian(field, coordinates)/(2*mass)


def build_proof():
    coordinates, t, momentum, m, hbar, amplitude = symbols()
    p2 = sum(p*p for p in momentum)
    energy = p2/(2*m)
    phase = (sum(p*q for p, q in zip(momentum, coordinates))-energy*t)/hbar
    wave = amplitude*sp.exp(sp.I*phase)
    return dict(wave=wave, energy=energy, gradient=sp.Tuple(*(sp.I*p*wave/hbar for p in momentum)),
                laplacian=-p2*wave/hbar**2, time_derivative=-sp.I*energy*wave/hbar,
                hamiltonian_action=energy*wave)


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed reconstruction changed')
    coordinates, t, momentum, m, hbar, amplitude = symbols()
    wave, energy = proof['wave'], proof['energy']
    p2 = sum(p*p for p in momentum)
    checks = {f'gradient_{i}': sp.diff(wave, q)-proof['gradient'][i]
              for i, q in enumerate(coordinates)}
    checks.update(time_derivative=sp.diff(wave, t)-proof['time_derivative'],
                  laplacian=laplacian(wave, coordinates)-proof['laplacian'],
                  kinetic_hamiltonian=-hbar**2*laplacian(wave, coordinates)/(2*m)-proof['hamiltonian_action'],
                  energy_time_generator=sp.I*hbar*sp.diff(wave, t)-energy*wave,
                  free_equation=residual(wave, coordinates, t, m, hbar),
                  zero_momentum=wave.subs(dict.fromkeys(momentum, 0))-amplitude,
                  phase_norm=sp.expand_complex(wave*sp.conjugate(wave)-amplitude*sp.conjugate(amplitude)))
    h, wavelength, frequency, k, omega = sp.symbols('h wavelength frequency k omega', positive=True)
    checks.update(wavelength_reciprocal=(k/(2*sp.pi)-1/wavelength).subs(k, 2*sp.pi/wavelength),
                  de_broglie=(h/wavelength-h*k/(2*sp.pi)).subs(k, 2*sp.pi/wavelength),
                  planck_frequency=(h*frequency-(h/(2*sp.pi))*omega).subs(omega, 2*sp.pi*frequency))
    # Operator linearity establishes any finite sum of separately valid modes;
    # it does not make arbitrary functions solutions.
    c1, c2 = sp.symbols('c1 c2')
    u = sp.Function('u')(*coordinates, t)
    v = sp.Function('v')(*coordinates, t)
    checks['operator_linearity'] = residual(c1*u+c2*v, coordinates, t, m, hbar)-c1*residual(u, coordinates, t, m, hbar)-c2*residual(v, coordinates, t, m, hbar)
    other = wave.subs(dict(zip(momentum, (2, -3, 1))))
    checks['two_mode_superposition'] = residual(wave+other, coordinates, t, m, hbar)
    # The momentum probability current gives a separate continuity check.
    density = wave*sp.conjugate(wave)
    current = [hbar*(sp.conjugate(wave)*sp.diff(wave, q)-wave*sp.diff(sp.conjugate(wave), q))/(2*m*sp.I)
               for q in coordinates]
    checks['probability_continuity'] = sp.diff(density, t)+sum(sp.diff(j, q) for j, q in zip(current, coordinates))
    for name, value in checks.items():
        if sp.simplify(value) != 0:
            raise ValueError('Free-wave check failed: '+name)
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=29, source_ast_parity=False,
                reconstructed_checks={k: True for k in checks},
                assumptions=['Three-dimensional Cartesian coordinates and real time.',
                             'Positive constant mass and reduced Planck constant; constant real momentum components.',
                             'Constant finite complex amplitude; energy p dot p/(2*m), no potential.',
                             'Free nonrelativistic scalar Schrödinger dynamics, complex convention exp(i*(p dot r-E*t)/hbar).',
                             'Scalar wavelength prerequisites require nonzero momentum; the final vector mode includes zero momentum.'],
                limitations=['Plane-wave consistency is not a derivation of quantum mechanics from classical laws.',
                             'Linearity extends solutions to finite sums of valid modes, not arbitrary wavefunctions.',
                             'Infinite superpositions require separate convergence and differentiation justification.',
                             'Nonzero plane waves are not square-integrable on infinite space; no normalized-state claim.',
                             'No external potential, relativistic, spin or boundary-condition model.'])
