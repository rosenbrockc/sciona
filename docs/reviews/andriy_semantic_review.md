# Andriy automated Tier 3 review

The proposed publication scope is the documented current-runtime implementation
of the three Andriy model branches, with the 25 exact provider versions recorded
in `andriy_provider_closure.json`. It is not certification of the historical
MATLAB/R environment, predictive accuracy, or the complete eleven-model solution.
Only Tier 1 requires human certification; this review targets Tier 3.

The current implementation preserves these source distinctions:

- CSP candidates are the ordered positive-file enumeration. Selection skips
  the first two entries, then uses sequence 6 and sequence 1. These are separate
  from the positive and negative model-training groups.
- Caller-supplied P, P1 and I memberships determine training order. Positive
  rows precede negative rows, preserving the source R class-column convention.
  Membership and ordering are not inferred from filenames or sequence numbers.
- CSP fitting and normalization remain separate for each of three populations.
  Normalization uses retained training and valid prediction rows together;
  the prediction validity mask is computed before normalization.
- The feature layout retains 19 complete windows per clip and 1965 ordered
  columns. Clip-level imputation and selected logarithms precede population
  assembly. Data-dependent training-row removal remains explicit.
- SVM masks invalid windows before segment maxima. XGBoost and GLM maximize
  all windows and zero only wholly invalid segments before their rank steps.
  Rank operations preserve source population concatenation and average ties.

The compatibility adaptations are part of the proposed contract. They include
explicit resampling FIR and Welch settings, documented estimated-state AR loss,
current numerical filtering/CSP behavior, current pinned R engines, numeric
conversion of source-shaped labels, and explicit one-based XGBoost feature
names. Historical numerical identity remains unproven. Degenerate CSP inputs,
nonfinite model training arrays and insufficient selector feature counts are
rejected; those restrictions must remain visible in the published descriptions.

Evidence is separated by what it establishes. The population source reference
checks preprocessing through normalized arrays. The three R reference reports
check model calculations independently. The model graph report checks exact
serialized-runner output routing. The 202-test aggregate covers contracts,
helpers, witnesses and routing. Catalog contract verification checks the 104
ports in each representation and all 21 pinned provider dependencies. None of
these alone proves full raw execution or predictive quality.

The six-training-clip raw case failed the source GLM selector gate. Its finite,
nonconstant inputs did not guarantee enough used features. That failure is
retained in `andriy_raw_small_sample_failure.json`; thresholds and model settings
were not weakened. The 28-training-clip case completed all four graph nodes
and six finite outputs with expected shapes in 1241 seconds. SVM score spread
was only 0.0002438593 on random labels; the case establishes execution, not
discrimination. Separate source-model reference fixtures provide nondegenerate
model comparisons. The full raw report and 98-file integrity check now pass.

The reviewed implementation is acceptable for automated Tier 3 publication
with these limits. This document does not mutate catalog state. The approval
transaction must recheck exact code, test and evidence hashes, raw-run evidence,
catalog bindings and provider versions, then verify served status. The full
eleven-model ensemble and remaining competition/physics backlog remain open.
