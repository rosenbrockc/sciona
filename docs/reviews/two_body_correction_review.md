# Circular two-body period automated Tier 3 review

The execution realization computes T=2*pi*sqrt(r^3/[G(m1+m2)]) for positive
finite SI inputs. r is the two-body separation, not either barycentric radius.
This is the isolated Newtonian two-point-mass circular-orbit model. The caller
must establish applicability and units; no orbit classifier, relativistic,
extended-body or external-perturbation correction is implemented.

The original 13-step graph remains draft. The separately corrected proof fixes
three reversed substitution labels, explicitly divides by positive second mass,
and uses the separation premise to justify a rational rewrite. All 13 selected
source equations and six premises agree under the explicit positive-real
mapping, but literal source-AST and inference-rule parity are not claimed.
The initial centripetal-force statement is a universally instantiated schema,
not renaming of a fixed-object fact. Exact source identity pdg0003141 is reviewed
as mathematical pi; the numerical implementation uses binary64 math.pi.

Provider validation rejects nonpositive, nonfinite, boolean, complex, nonnumeric,
empty, mismatched-shape or nonscalar-G inputs. Output preserves shape; no
broadcasting or input mutation. Per-element 100-digit Decimal arithmetic avoids
intermediate mass-sum/cube overflow and underflow; results rounding to zero or
infinity are rejected, while positive subnormals are permitted with reduced
relative precision. No throughput or universal correct-rounding claim.

Evidence: 68 tests pass. Five fresh full-runner cases evaluate 208 synthetic
inputs and inspect saved outputs after graph serialization round trip. Maximum
error is one float64 ULP against independent 180-digit mpmath with exact pi.
Tests include independently mutated proof premises/intermediates, exact positive
witnesses, singularity rejection, dynamic range, shape and mass symmetry.

Reference: OpenStax Calculus Volume 3, section 3.4, equation 3.30,
https://openstax.org/books/calculus-volume-3/pages/3-4-motion-in-space
supports the total-mass period formula. Circular separation semantics follow
from the explicitly reviewed premises. This review supports automated Tier 3
only, with no Tier 1 human-certification or Tier 2 usage-evidence claim.
