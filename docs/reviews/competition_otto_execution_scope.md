# Otto execution scope for automated Tier 3 review

This candidate is an independent implementation of the documented first-place
Otto stacking method. It is not approved yet. The proposed designation is
Tier 3 Community; it makes no claim to Tier 1 human review, Tier 2 usage
qualification, historical winner accuracy, or exact reproduction.

The source intake version is `dc4969f7-e5a4-5644-a9d0-fa10493458ed`, with content
hash `13be8fd78b963daf9fd05fb5758e1d36fa98d4ed8cfdde85eb829c94f73af30a`.
Any published realization must preserve a mandatory dependency on that exact
source. The original intake remains unchanged.

The implementation includes all 33 enumerated first-level entries, shared
five-fold out-of-fold predictions and full-reference query refits, seven
shared supplemental feature blocks, and raw features for the neural meta
learner. Four-fold pooled log loss selects controls separately for XGBoost,
Lasagne and AdaBoost with ExtraTrees. Selected controls are refit on all meta
training rows with 250, 600 and 250 seed-varied models respectively. Bag means
enter the documented geometric/arithmetic blend, followed by normalization.
Meta selection losses are selection diagnostics, not an independent estimate
of generalization: upstream first-level OOF features are not rebuilt inside
each meta selection fold.

Explicit limits and implementation choices:

- The t-SNE entry supplies three coordinates plus two cluster features. This
  yields 293 first-level columns, not the historically reported 297. No
  additional prediction columns are invented to reconcile the discrepancy.
- Metric counts, cluster counts, interaction subsets, network widths, epochs,
  boosting controls and their candidate grids are explicit caller choices.
  Synthetic validation does not establish the historical supplemental width,
  hyperparameter search space or training volume.
- The two 120-model neural families execute on the CPU. Historical GPU
  execution and equivalence remain unqualified. Two/three hidden layers are
  the explicit interpretation of the ambiguous layer descriptions.
- Modern Python XGBoost and H2O interfaces replace historical R interfaces.
  AdaBoost uses explicit SAMME controls. No equivalence with a historical
  SAMME.R configuration is asserted.
- Native RSofia uses stable sigmoid conversion of linear predictions. The
  interaction variant scales the assembled feature block. The ROC variant
  uses explicit log scope, retaining signed embedding coordinates when only
  nonnegative blocks are transformed.
- t-SNE uses a fixed population that may include query features, without
  query labels. It cannot embed previously unseen identities. The common
  embedding and unsupervised clustering are not claimed to be inductive.
- Runtime validation uses synthetic inputs only. Native R, Java, libFM and
  patched Theano/Lasagne dependencies require a recorded environment and
  retained notices before publication. Clean installation, other platforms,
  resource scaling and production deployment remain unqualified.

Publication still requires the actual serialized graph/provider execution,
complete environment and license closure, hash-bound automated review,
negative gate checks, rollback verification, idempotent publication and
served-artifact execution. Passing component or full-pipeline tests alone
does not change catalog approval.
