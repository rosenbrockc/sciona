# Periodic-wave relations: source and proof review

Source version 0873a091-c6f6-56a1-a80d-976b59b95a8c, hash
aee08ee2103f0721af8363fcb34fe5b7f5ad172697c3824800d73473370686cf.

Status: source, proof, runtime and serialized execution review passed. Tier 3
publication requires fresh evidence equality and catalog transaction gates.

Six source steps have sixteen bindings and ten distinct equations. Four stored
snapshots are freshly audited; six missing distinct equations are recovered
from the expression file pinned by original ingestion. The review checks exact
LaTeX, source equation sides under explicit symbolic interpretation, per-step
rules and bindings, original feeds, source dimensions and all file pins.

Step 4 multiplies T=1/f by a different source identity pdg0006235, whose label
is also f but whose dimensions are dimensionless. Cancellation to Tf=1 instead
requires the frequency identity pdg0004201, with dimensions T-1. The reconstruction
explicitly corrects that feed. Arbitrary symbols sharing a name are not equated.
The exact source pi identity is interpreted as mathematical pi using its pinned
constant declaration. Original source remains draft and unchanged.

Stored step numbers do not follow dependencies: step 1 uses the output of
step 6. The proof replays 4,5,6,1,2,3. It obtains Tf=1, f=1/T, omega=2*pi/T,
k=2*pi/(vT), lambda=vT, and k=2*pi/lambda. All ten source equations match the
reviewed equations after the exact identity mapping. The four root relations
are T=1/f, lambda=v/f, omega=2*pi*f and k=omega/v.

The domain is a single positive-frequency periodic wave with positive phase-speed
magnitude. Frequency uses cycles/s, period seconds, wavelength meters, angular
frequency rad/s and angular wavenumber rad/m. The result is a magnitude, not a
signed wavevector. Phase velocity is not automatically group velocity. No
dispersion relation or broadband inference is established.

All nine proof tests pass: dependency replay and phase-cycle checks, six corrupt
transitions, the unrelated-f cancellation error, and angular wavenumber versus
spatial cycles. Both omega*T and k*lambda equal 2*pi. Fixtures are synthetic
mathematics only. No real or templated datasets are involved.

The completed runtime review follows. Publication still requires the fresh
validation and catalog gates recorded by the promotion script.


Runtime review: the new provider returns T, wavelength, omega and angular k from
positive frequency and phase-speed magnitude. The existing period-to-frequency
provider was inspected; it covers only that conversion. Full source relations
require all four outputs. Rational quantities use exact Fraction arithmetic;
angular quantities use isolated 450-digit mpmath. No rounded output is reused.
All outputs must be finite positive representable float64; subnormals accepted.
32 provider/proof tests pass including independent Decimal2500 and mpmath1000
oracles, all four units/conventions, extrema, invalid input/output domains and
concurrent precision isolation. Six serialized runner cases cover 29 synthetic
states with zero ULP error for all four outputs. Provider/proof/validator/test/
parser/codec/runner evidence is hash-bound. No real or templated data used.
Reference: [OpenStax Mathematics of Waves](https://openstax.org/books/university-physics-volume-1/pages/16-2-mathematics-of-waves).
No Tier 1 or Tier 2 claim. Original draft remains unchanged.
