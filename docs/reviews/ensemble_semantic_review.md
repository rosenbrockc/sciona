# Full ensemble automated Tier 3 review — acceptable with limits

The proposed scope is the current-runtime reconstruction of all eleven source
branches and final identity-aligned rank blend. The graph contains 155 nodes,
523 edges and 18 explicit root inputs. Its provider closure contains 59 exact
versions: 52 already served providers and seven new adapter drafts. Human
certification is required only for Tier 1; this review targets Tier 3.

The graph preserves separate caller-supplied training groups for Alex, Feng
and Andriy. It does not infer source-safe membership or assume identical
training inclusion rules. Alex receives channels/samples at 400 Hz; Feng
receives samples/channels source-duration clips. Andriy retains its explicit
CSP candidates, sequence selection and P/P1/I groups. These boundaries and
all source configuration nodes remain unchanged when composing branches.

The eight Alex/Feng models run separately for each population. Their paired
scores and opaque IDs are concatenated without sorting or ranking. The three
Andriy branches retain their own population handling and branch-specific
validity/ranking rules. The identity adapter checks each Andriy population's
clip count rather than only the total count.

The packer fixes the source's eleven-model order. Each family must preserve
the common prediction order established by its selectors. The final blend
ranks each complete model vector before selecting output IDs, applies eleven
equal weights and adds the supplied baseline without clipping or reranking.
Extra prediction IDs affect rank denominators, while missing or duplicate IDs
are rejected. Opaque IDs must correspond to the same physical prediction clips
across families; the adapters cannot establish that semantic mapping from
numeric identifiers alone.

Evidence has distinct scopes: 24 tests cover contracts, adapters, graph
serialization, callable boundaries and actual witness propagation. Three
source-blend comparisons cover reordering, extra IDs, ties and nonzero
baselines. Existing component reference reports establish upstream numerical
behavior with declared modern-runtime adaptations. Exact catalog versions,
interfaces and bindings establish identity consistency, not numerical success.

The successful full raw run used shared physical synthetic prediction clips,
reversed Feng prediction order and a separately permuted requested output
order. It executed all 155 nodes without cached outputs and exactly matched the final
blend against pinned source code using the actual model outputs. This case
uses random class assignments and cannot establish predictive quality or
historical-engine equivalence. All eleven score vectors were finite and nonconstant; the final score spread
was 0.287878787878788. Execution completed in 1299.51 seconds.

The wiring module imports a private group-length validator from the companion
partition module. Shared source-file hashes cover that code dependency; it is
not a call to another registered atom. Final approval must recheck those
hashes along with exact served dependencies, adapter interfaces, all 155
bindings, numerical execution evidence and the catalog graph snapshot.

The reviewed implementation is acceptable for automated Tier 3 publication,
subject to the atomic catalog identity and serving checks in the approval script.
The broader competition and physics backlog remains outside this ensemble review.
