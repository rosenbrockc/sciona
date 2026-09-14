# Constant-force work–energy theorem: source and proof review

Source version 9bed89a4-8da7-59e4-a67a-1bdc0a1fffe0, hash
8f8447851adca1da1db0d209a0c61b3ac455d5c6bfe54e6e969b4f5cfd4dace5.

Status: source, proof, implementation and serialized execution review passed.
Tier 3 publication requires fresh evidence equality and catalog transaction gates.

Seven steps, eighteen bindings and twelve distinct equations are reviewed.
Ten stored snapshots are freshly audited; two missing equations (Newton's law
5345738321 and the constant-acceleration displacement formula 5611024898) are
recovered from the original ingestion's pinned expression file. Exact LaTeX,
per-step rules/bindings, original feeds, source identities and dimensions are
checked. Work, force-displacement and kinetic-energy dimensions agree.

The source integral mixes an indefinite work integral with definite spatial
limits and omits the work reference. The reconstruction defines accumulated net
work as zero at the initial displacement, uses separate integration dummies,
and integrates from zero to W and zero to signed displacement x. Constant force
then gives W=Fx. An arbitrary indefinite antiderivative instead contains a
constant and cannot be equated to Fx without a reference convention.

The source's three-variable rename feeds use the position identity where d is
required and use the final-speed identity twice, once for the initial speed.
The corrected simultaneous mapping is d->x, v->v2, v0->v1, using their exact
source identities. No blanket same-name substitution or literal AST parity is
claimed. Original source remains draft and unchanged.

The seven reconstructed steps integrate the work differential, evaluate the
definite integrals, substitute F=ma, rename the kinematic endpoints, substitute
displacement, cancel nonzero acceleration and replace the two kinetic-energy
definitions. The result is W=KE2-KE1, with KEj=m*vj^2/2.

The divided-acceleration route requires a!=0. An independent trajectory identity
uses x=v1*t+a*t^2/2 and v2=v1+a*t without division. It verifies the result for
constant signed acceleration, including a=0, when velocity is unchanged and net
work is zero. Signed displacement and velocities permit deceleration and reversal.

Scope is classical constant positive mass, one-dimensional motion and constant
net force. The result describes net translational work. It is not work by an
arbitrary individual force, dissipated heat, rotational energy, relativistic
energy or a variable-mass system. The caller must supply consistent endpoints;
the theorem is not a trajectory reconstruction algorithm.

All 14 proof tests pass: seven corrupt transitions, independent trajectories
including deceleration/reversal/zero acceleration, and the integration-reference
counterexample. Fixtures are synthetic mathematics only.

The registered work_energy provider accepts positive mass and signed initial/final
velocities with identical nonempty finite shapes, including scalars. It returns
both kinetic energies and signed work. Exact rational arithmetic on float64
inputs precedes independent rounding; work does not subtract rounded outputs.
A dedicated regression demonstrates that those two calculations differ.
All outputs must be representable; zeros/subnormals accepted, nonzero underflow
and overflow rejected. Input shape and values preserved, no broadcasting.

All 40 combined tests pass. Six serialized runner cases cover 29 synthetic states,
including negative work, reversal, zero motion, large intermediate squares and
near-equal energies. All three outputs have zero ULP error against an independent
Decimal2500 oracle using the factored difference (v2-v1)*(v2+v1). Source/provider/
proof/validator/test/codec/runner evidence is hash-bound. No real or templated data.
[OpenStax Work-Energy Theorem](https://openstax.org/books/university-physics-volume-1/pages/7-3-work-energy-theorem)
supports the net-work interpretation. No Tier 1 or Tier 2 claim is made.
