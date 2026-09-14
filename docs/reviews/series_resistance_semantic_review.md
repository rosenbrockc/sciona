# Semantic review: two-component series resistance

**Decision requested:** accept or reject the equations and domain below for the
source derivation and its numerical realization. This is a semantic review;
licensing and release availability still have separate unresolved gates.

The seven canonical source-expression versions now pass the project's automated
import, parsing, dimensional, symbolic-validation, and source-verification gates.
Their human-review gate and publication gate remain unsatisfied. A successful
review must identify the reviewer and review time and cover the validity bounds,
as required by `sciona/physics_ingest/review.py::_human_reviewed_gate`.

## Equations and domain

Symbols below are conventional presentation names. Voltage is in volts, current
in amperes, and resistance in ohms.

| Equation | Role |
| --- | --- |
| V = I R | General ohmic-component premise |
| V₁ = I R₁ | First series component |
| V₂ = I R₂ | Second series component |
| Vₜ = I Rₜ | Equivalent-component premise |
| Vₜ = V₁ + V₂ | Series voltage addition |
| I Rₜ = I R₁ + I R₂ | Substituted voltage balance |
| Rₜ = R₁ + R₂ | Terminal result |

The model requires lumped ohmic components in one series branch, a common branch
current, additive voltage drops, and nonnegative resistances. It makes no claim
for nonlinear components or other circuit topologies. The recorded derivation
requires **I ≠ 0** at its division step. The numerical realization retains that
guard even though the terminal sum itself is independent of current.

The executable interface takes `resistance_a`, `resistance_b`, and `current` and
returns `equivalent_resistance`. It uses float64 NumPy broadcasting, rejects
nonfinite or non-real inputs, negative resistance, zero current, incompatible
shapes, and arithmetic overflow. These numerical checks cannot establish whether
a physical device satisfies the model assumptions.

Scientific support: [OpenStax, University Physics Volume 2, §10.2](https://openstax.org/books/university-physics-volume-2/pages/10-2-resistors-in-series-and-parallel).
The section supports the common series current, voltage addition, Ohm's law, and
sum-of-resistances result. Reference association was automated; no human graph
approval has been recorded.

## Evidence available

- Source files match ingestion content pins. All seven stored expressions match
  their upstream ASTs after source-defined scalar mapping and pass SI checks.
- The four-step derivation replays using computed intermediate equations and
  canonical edges. Its necessary nonzero-current condition is stored for review.
- Each stored expression passed 256 consistent synthetic circuit scenarios and
  detected 256 deliberately inconsistent scenarios. Maximum absolute residual
  was 1.42e-14, against a 1e-10 test tolerance. These are finite tests, not an
  exhaustive proof or experimental validation of physical components.
- The generated terminal runtime passed a circuit linear-system reference and
  permutation/current-scaling checks. The registered primitive ran through the
  real CDG runner on 256 cases; relative discrepancy was approximately 2.22e-16.
- A locally built wheel contains the exact implementation and passes isolated
  installation tests with SymPy imports blocked. No release availability is
  claimed.
- The numerical execution artifact remains draft, not latest, and unpublishable.
  It retains a mandatory pin to the original source derivation and an exact
  primitive version binding. No source dependency review was bypassed.

## Review response

Please provide the reviewer name/identity, acceptance or rejection of this scope,
and any required corrections or additional physical bounds. A review response
will be attached only to the specific versions covered by the verified evidence.
It will not authorize publication of other competition or physics CDGs.

The provider's license manifest still declares `NOASSERTION` / unknown. The
separate license/private-use policy question remains open.
