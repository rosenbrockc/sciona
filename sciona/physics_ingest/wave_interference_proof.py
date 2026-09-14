"""Two scalar-wave intensities: fixed phase versus an explicit ensemble mean."""
import sympy as sp

SOURCE_VERSION = '97e986ef-3194-5064-a050-7a65ac13ca34'
SOURCE_HASH = 'c937bbfa09c8e33df807bfbba7512a4cdf2975da7d63aea708af173b95680635'


def build_proof():
    a, b = sp.symbols('a b', nonnegative=True)
    theta, phi, delta = sp.symbols('theta phi delta', real=True)
    A, B = a*sp.exp(sp.I*theta), b*sp.exp(sp.I*phi)
    intensity = a*a+b*b+2*a*b*sp.cos(delta)
    return dict(amplitude_a=A, amplitude_b=B,
                product=(A+B)*sp.conjugate(A+B), intensity=intensity,
                stable_intensity=(a-b)**2+4*a*b*sp.cos(delta/2)**2,
                incoherent_mean=a*a+b*b, constructive=(a+b)**2,
                destructive=(a-b)**2, interference=2*a*b*sp.cos(delta),
                ratio=intensity/(a*a+b*b))


def verify_proof(proof):
    if proof != build_proof():
        raise ValueError('Reviewed reconstruction changed')
    a, b = sp.symbols('a b', nonnegative=True)
    theta, phi, delta = sp.symbols('theta phi delta', real=True)
    A, B = proof['amplitude_a'], proof['amplitude_b']
    intensity = proof['intensity']
    checks = {
        'conjugate_a': sp.conjugate(A)-a*sp.exp(-sp.I*theta),
        'conjugate_b': sp.conjugate(B)-b*sp.exp(-sp.I*phi),
        'modulus_a': A*sp.conjugate(A)-a*a,
        'modulus_b': B*sp.conjugate(B)-b*b,
        'expanded_product': proof['product']-(A*sp.conjugate(A)+B*sp.conjugate(B)+A*sp.conjugate(B)+B*sp.conjugate(A)),
        'cross_terms': A*sp.conjugate(B)+B*sp.conjugate(A)-2*a*b*sp.cos(theta-phi),
        'intensity_definition': proof['product']-intensity.subs(delta, theta-phi),
        'stable_positive_form': proof['stable_intensity']-intensity,
        'constructive_phase': intensity.subs(delta, 0)-proof['constructive'],
        'destructive_phase': intensity.subs(delta, sp.pi)-proof['destructive'],
        'uniform_phase_cross_mean': sp.integrate(proof['interference'], (delta, -sp.pi, sp.pi))/(2*sp.pi),
        'uniform_phase_intensity_mean': sp.integrate(intensity, (delta, -sp.pi, sp.pi))/(2*sp.pi)-proof['incoherent_mean'],
        'equal_amplitude_constructive': proof['constructive'].subs(b, a)-4*a*a,
        'equal_amplitude_incoherent': proof['incoherent_mean'].subs(b, a)-2*a*a,
        'equal_amplitude_ratio': proof['ratio'].subs({b: a, delta: 0}, simultaneous=True)-2,
        'lower_bound_gap': intensity-proof['destructive']-4*a*b*sp.cos(delta/2)**2,
        'upper_bound_gap': proof['constructive']-intensity-4*a*b*sp.sin(delta/2)**2,
        'constructive_ratio_bound': 2*(a*a+b*b)-proof['constructive']-(a-b)**2,
        'single_wave': intensity.subs(b, 0)-a*a,
    }
    for name, residual in checks.items():
        if sp.simplify(sp.expand_complex(residual)) != 0:
            raise ValueError('Interference identity failed: '+name)
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_steps_reviewed=27, source_ast_parity=False,
                reconstructed_checks={k: True for k in checks},
                assumptions=['Scalar complex amplitudes in the same polarization/mode and common intensity normalization.',
                             'Real nonnegative fixed amplitudes a,b and finite real relative phase delta.',
                             'Fixed-phase intensity is normalized squared magnitude, not an incoherent average.',
                             'Incoherent mean assumes vanishing averaged cosine with fixed amplitudes; uniform relative phase is sufficient.',
                             'Ratio requires a^2+b^2>0; factor two further requires a=b>0 and delta=0 modulo 2*pi.'],
                limitations=['Coherent waves need not be constructive; anti-phase equal amplitudes cancel.',
                             'The averaged cosine condition is not an instantaneous cosine=0 constraint.',
                             'No arbitrary polarization, spatial propagation, stochastic-amplitude or partial-coherence model.',
                             'Amplitudes and intensity are normalized; conversion to physical intensity requires a common external scale.',
                             'All-zero amplitudes have zero intensities but an undefined ratio and are excluded from the combined runtime contract.'])
