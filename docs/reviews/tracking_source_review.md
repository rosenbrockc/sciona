# Automated Tier 3 review: original analytical tracking

The executable realization follows the pinned original implementation through
detector construction, first/second-hit seeding, repeated helix extension,
fitting, pairing, ranking, redundant-candidate pruning and final commitment.
It supersedes neither the original intake graph nor its individual helper
bindings; its mandatory source dependency preserves that provenance.

Twelve implementation/license resources are retained byte-for-byte, verified
against a manifest whose hash is fixed in the loader. Each invocation loads
fresh private modules; source imports do not enter global `sys.modules`.
Three reviewed `DataFrame.as_matrix` calls use an explicit conversion helper.
The original event loader receives an in-memory frame after float32 coordinate
conversion, preserving the original public loader's numerical semantics.
Final output is captured in memory. Input arrays remain unchanged.

The wrapper uses the source analytical defaults with two explicit changes:
quadratic ranking coefficient 1 for small inputs and redundant-candidate pruning
from the first extension iteration. Extension count and commitment options are
bounded caller inputs. The default extension count is eleven; all default ports
were exercised through the real CDG runner as well as explicit single-round and
three-round settings.

Thirty-five tests cover exact membership of eight generated helices, input
preservation, concurrent invocation isolation, independent source defaults,
malformed matrices/options/identifiers, coordinate overflow, and fail-closed
source/manifest integrity. Three serialized CDG runner cases recover the same
eight helices with exact membership among 160 synthetic observations. No actual
dataset records, templates, or metadata were used.

This review does not establish actual-event accuracy, throughput, learned layer
calibration, cell features, scoring, hyperparameter tuning, or nonphysical
postprocessing. Source geometry assumptions remain: compatible finite cylinders,
seven cap radial clusters with three gaps, monotone cylinder coverage and
sufficient occupied neighboring layers. Unsupported geometry fails rather than
receiving a substituted tracking algorithm. Upstream escape-sequence warnings
and environment cache/core-detection warnings are nonfatal in the tested runtime.

The provider closure and compatibility helper hashes are part of the evidence;
deployments must include them and the declared tracking dependencies. This is
automated Tier 3 evidence, not human-reviewed Tier 1 certification.
