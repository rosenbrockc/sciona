# Two-wave interference: reconstructed execution review

Source version: 97e986ef-3194-5064-a050-7a65ac13ca34.
Source graph hash: c937bbfa09c8e33df807bfbba7512a4cdf2975da7d63aea708af173b95680635.

The source contains 27 steps, 66 bindings, and 34 unique stored records,
including a phase feed incorrectly bound as an equation. All public source
fields, inference rules, feeds, bindings and file pins have been freshly checked.
The source remains draft. This review concerns a reconstructed executable model,
not a literal replay or validation of all 27 source steps.

Corrections restore lost conjugates and blank product expressions, replace the
imaginary-unit symbol with mathematical i, and repair malformed nested absolute
values in the cross terms. Step12 binds a theta feed instead of the polar-Z
equation and omits the phase replacement. Step16 lacks the A-conjugate and
expanded-intensity bindings needed for four substitutions. Several variable feeds
are reversed or incomplete. The cosine substitution must replace x by theta-phi
on both sides; the source leaves cos(x) unchanged. The source's averaged cosine
is not the instantaneous equation cos(theta-phi)=0 encoded in its AST.

The model uses normalized nonnegative magnitudes a,b of scalar waves in a common
mode/polarization, with real relative phase delta and common intensity scale.
It evaluates I=a²+b²+2ab*cos(delta), the incoherent mean a²+b², extrema (a+b)²
and (a-b)², the signed interference term, and I/(a²+b²). The mean assumes fixed
amplitudes and vanishing averaged cosine; uniform phase is sufficient. Changing
amplitudes correlated with phase invalidates that simple averaging argument.
Both magnitudes zero are excluded from this combined interface because the ratio
is undefined. Destructive cancellation with nonzero magnitudes has a defined zero
ratio. The factor-two result requires equal nonzero magnitudes and constructive
phase alignment. Coherence alone does not imply constructive interference.

These distinctions agree with the discussion of fixed and random phase
relationships and constructive/destructive addition in
[OpenStax University Physics Volume 3](https://openstax.org/books/university-physics-volume-3/pages/3-1-youngs-double-slit-interference).
This implementation does not model propagation, arbitrary polarization,
stochastic amplitudes or partial coherence. Physical intensity calibration
requires a common external scale; all nine graph ports are dimensionless.

Nineteen symbolic checks cover complex products, cross terms, normalized
intensity, a positive evaluation formula, uniform-phase averaging, extrema,
the equal-amplitude ratio, bounds and the single-wave limit. Synthetic tests
include counterexamples to treating coherence as constructive, treating the
ensemble mean as an instantaneous condition, and ignoring amplitude/phase
correlation. One initial test compared unevaluated SymPy expressions structurally;
it now expands both mathematical products before checking their values.

The provider uses local 1200-digit arithmetic and the nonnegative expression
(a-b)²+4ab*cos(delta/2)². It uses converted float64 phases literally, including
large values; float64 pi is not snapped to exact pi. Outputs are rounded
independently, accepting exact zero and subnormals and rejecting nonzero
underflow or overflow. There is no universal correct-rounding claim or exact
rounded-output sum/ratio guarantee.

All 31 tests passed. Six serialized-runner cases cover 29 synthetic states with
zero ULP difference from a 2000-digit complex-field reference. Cases include
constructive alignment, near-destructive cancellation, unequal magnitudes,
very large positive/negative phase, single-wave and subnormal intensity.
The reference obtains incoherent mean by averaging opposite-phase fields.
Retained reports bind the provider, tests, proof, source audit and runner code.
Approval is automated Tier 3, with no Tier 1 designation or literal AST parity
claim. Catalog promotion and independent verification are recorded separately.
