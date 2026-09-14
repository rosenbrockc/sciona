# Circular-path speed source reconstruction

Source61588847-3d80-5c3a-b4b4-21f10988a6bc, version
ec76168f-50f7-5a81-b2d7-0be499024080 remains draft. Fresh audit verifies eight
steps, nineteen bindings, ten stored equations and two missing equations recovered
from pinned original public source (5426308937 speed and6785303857 circumference).
All three source file pins, exact symbolic fields and decoded LaTeX checked.

The source estimates Earth orbital speed using v=2*pi*r/T. Its scientific notation
and units are partly absent from symbolic fields: radius1.496e8km becomes1.496,
period3.16e7s becomes3, and substituted forms drop powers of ten/radius. The unit
conversion feed stores365 rather than365*24*60*60. Reconstruct the physical
quantities from reviewed LaTeX and use exact pi and integer conversion factors.

A365day period is31536000s. Keeping it exact with radius149600000km gives
29.806079463282158072km/s, rounding to29.8. In contrast, the displayed rounded
period31600000s gives29.745712720065384081km/s, rounding to29.7 at one decimal.
Thus the source exact-equality chain does not justify its displayed endpoint.
Retain exact conversion and mark the final29.8 as approximate.

Scope: average speed around a circular path, instantaneous speed only if motion
is uniform. The radius/year are educational approximations, not an ephemeris.
Earth is not on an exactly circular orbit; no eccentric-orbit or gravitational
model is claimed. NASA describes its non-circular orbit and varying speed:
https://science.nasa.gov/learn/basics-of-space-flight/chapter2-1/

Runtime implementation, serialized runner and Tier3 promotion remain pending.
Implement general positive radius/period calculation with unit-consistent SI
inputs, plus the full reviewed Earth example in synthetic execution evidence.
Preserve circumference and speed computation with sufficient precision and
independent rounding/representability; do not hard-code29.8 as the runtime result.

Runtime completed:34 combined tests and6 serialized runner cases covering29
synthetic states passed. Circumference and speed independently rounded from450
digit local-context arithmetic;1000digit reference agrees exactly on all cases.
Inputs positive finite matching arrays in SI, outputs positive finite representable.
No broadcasting, no rounded circumference reuse, subnormals accepted, underflow
and overflow rejected. Full educational Earth example and prematurely rounded
period contrast included. Tier3 transaction outcome recorded separately.
