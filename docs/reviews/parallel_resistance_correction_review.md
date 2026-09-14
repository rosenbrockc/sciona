# Two parallel resistors: source and domain review

Source121cbf9d-f0e5-5833-a02e-af773a519f2a, version
90400aa6-60ab-58d4-a7cd-295acde66d27 remains draft. Eight steps, nineteen
bindings and ten equations audited. Nine stored equations are present; the three
missing binding occurrences refer to one Ohms-law equation4087145886, recovered
from pinned original Cypher. Exact source fields, renaming/division feeds and
all three public source file pins verified. Nine electrical symbol dimensions
agree: voltage=current*resistance. No dimension correction needed.

Specify two ideal positive finite linear resistors connected across the same
nodes, with common signed voltage and currents oriented consistently. Ohms law
for each branch plus Kirchhoff current sum gives V/Rt=V/R1+V/R2. The source
cancels V, requiring a nonzero test voltage. Rt=R1*R2/(R1+R2) follows from the
conductance law. Extending the constitutive identity to arbitrary voltage gives
I_total*Rt=V including V=0 with all currents zero. A single zero-voltage reading
does not identify resistance through V/I_total=0/0.

Fourteen synthetic tests cover every corrupt step, zero/positive/negative voltage,
a series-network counterexample and zero-voltage identifiability. All pass.
No ideal short/open circuit, negative resistance, nonlinear or reactive impedance
scope. Runtime, serialized runner and Tier3 transaction remain pending.

Runtime should compute equivalent resistance, two signed branch currents and
signed total current from R1,R2,V. Use exact rational arithmetic on converted
float64 values and independently round each output to avoid overflow of R1*R2,
R1+R2, reciprocal overflow or reuse of rounded equivalent resistance. Require
positive finite resistances, finite signed voltage, identical nonempty shapes.
Exact zero currents valid; nonzero underflow and nonfinite outputs rejected.

Runtime completion:38 tests and6 serialized runner cases/29 synthetic states
passed. All four outputs agree exactly with independent2500digit Decimal
conductance-form oracle. Exact Fraction implementation avoids intermediate
product/sum/reciprocal overflow. Positive/negative/zero voltage, exchange symmetry,
invalid inputs and extreme resistances verified. Graph round trip and actual
runner output arrays checked. Source-stage pending statements above are historical;
final Tier3 transaction outcome is in separate catalog_verification report.

Reference: OpenStax University Physics Volume2,10.2 Resistors in Series and Parallel:
https://openstax.org/books/university-physics-volume-2/pages/10-2-resistors-in-series-and-parallel
