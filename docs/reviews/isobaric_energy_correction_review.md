# Isobaric internal-energy derivative: automated Tier 3 review

Source version: a8f28449-1e31-5c15-b514-b632020fc1a8.
Source content hash: af502248823bbbd0d4a43f09cd66173c7bdf82d448b3d793d90bfe727a3e7451.

The reviewed realization computes the local derivative
`(partial U/partial T)_p = Cv + pi_T V alpha`.
Five source transitions and thirteen bindings are checked against nine source
equations, fresh payload evidence, and pinned symbol and inference-rule files.
The original draft has incomplete derivative ASTs and remains unchanged.
This approval concerns the explicitly reconstructed execution graph; it does
not assert literal parity with those ASTs.

The source defines internal pressure as `(partial U/partial V)_T` but assigns
force dimensions `M1 L1 T-2`. Dividing the source energy dimensions by volume
instead yields pressure `M1 L-1 T-2`. The reconstruction makes that correction
explicit. Both terms of the result have energy per temperature dimensions.

The proof substitutes the two local derivative definitions into the total
differential, pulls it back to a fixed-pressure path parametrized by temperature,
substitutes `V alpha` for the path's volume derivative, and uses `dT/dT=1`.
It assumes differentiable U(T,V), a differentiable fixed-pressure path, positive
volume, and fixed composition and amount. All coefficients must describe the
same local state. Differential division is not finite-increment division.

The output is not Cp. For a simple compressible system Cp includes the additional
mechanical expansion term `p V alpha`; mechanical pressure p differs from pi_T.
This distinction is supported by [MIT 8.044, Spring 2013, problem set 4,
problem 1](https://ocw.mit.edu/courses/8-044-statistical-physics-i-spring-2013/f7b8910bfe299d327f60ae36e856718c_MIT8_044S13_ps4.pdf).

The registered provider accepts identical, nonempty finite real array shapes,
including scalar arrays, converts to float64, and rejects nonpositive volume.
Signed coefficients are supported without implying thermodynamic stability.
Exact rational arithmetic on converted values precedes one binary64 rounding.
Exact zeros and subnormals are supported; overflow and nonzero underflow to zero
are rejected. Inputs are preserved and no broadcasting is performed.

All 27 provider/proof tests pass, including an independently differentiated
synthetic energy along a constant-pressure path, the ideal-gas Cp distinction,
invalid inputs, extreme arithmetic and five corrupted proof transitions.
Six serialized full-runner cases cover 29 synthetic states with zero ULP error
against an independent 2500-digit Decimal calculation. Evidence hashes bind the
provider, proof, validators, tests, graph codec and runner. No templated or real
dataset contents are used.

Automated Tier 3 is the publication scope. No human-reviewed Tier 1, Tier 2
usage, material equation-of-state, stability, finite-temperature integration or
throughput claim is made. Publication requires fresh validation to match the
retained evidence, catalog graph round-trip validation, exact provider binding,
and preservation of original source provenance.
