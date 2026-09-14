The ten-stage competition graph remains draft. Active binding labels currently
overstate execution coverage: the pinned author code reveals missing computation
and behavior differences that cannot be repaired by renaming arguments.

Source: [author repository](https://github.com/lfz/DSB2017/tree/0ac3eb9f383bf0127c587e0502c59ac84d9ba6a2).
`competition_dsb_source_pins.json` pins ten explicitly selected source/license
files. The diagnostic reads only these public files and uses synthetic arrays;
it loads no real data, predictions, or checkpoints. No catalog mutation occurs.

| Stage | Source requirement and current gap |
| --- | --- |
| Lung mask with bone removal | Author preprocessing performs per-slice convex hulls, dilation, transformed-intensity masking, and boundary bone replacement. Current named callable returns only the segmentation mask. The template's “210 HU” description is also inaccurate: the code thresholds the transformed intensity at 210. |
| Volume split/combine | No named binding. The source implements edge-padded overlapping cubes and margin-stripped reconstruction. Porting must preserve tile ordering and output stride, explicitly handle Python 2 integer division, and repair the default grid-state mismatch (`nzhw` stored but `nz/nh/nw` read). |
| Coordinate-aware network | Current readiness reports no unique usable catalog binding for this stage. A real network with coordinate injection, matching weights/configuration, and tested runtime is required; a generic convolution helper is insufficient. |
| Anchor label mapping | Source distinguishes the selected target from all boxes, randomly chooses a positive candidate, and subsamples negatives in training. Current helper merges those inputs, chooses the first positive, and omits training negative subsampling. |
| Hard-negative mining | Current helper returns ranked indices only. Source additionally uses a batch-scaled selection count, weighted positive/negative binary cross entropy, and regression losses. |
| Size-aware oversampling | Source starts empty and adds 1/2/4 copies for strictly exceeded thresholds. Current helper also retains every original, including items below the minimum threshold, changing all multiplicities. The diagnostic executes the pinned original loop to demonstrate the difference. |
| Proposal sampling | Source recomputes softmax on remaining proposals each draw, floors probabilities at 1e-5, then renormalizes. Current helper performs one NumPy choice without the floor. Underflow can make its requested sample impossible. Source returns input order when the requested count covers all proposals. |
| Center feature extraction | Current helper implements center-cube maximum reduction. This is a component correspondence, not evidence for the surrounding trained network, classifier, or complete graph. |
| Noisy-OR aggregation | A formula helper is present. Full training requires the learned baseline initialized to -30 and a differentiable classifier path; formula correspondence alone does not verify these requirements. |
| Miss penalty | Current callable returns a NumPy scalar with an epsilon-modified logarithm. Before promotion, verify the training implementation's exact reduction, threshold, numerical policy, weighting, and gradient behavior. |

Next implementation work should preserve the complete ten-stage objective:
first supply source-faithful splitting/reconstruction and corrected sampling,
then validate preprocessing and label/loss behavior, and integrate the actual
coordinate-aware detector/classifier with synthetic execution and gradient
checks. These components must ultimately be tested as a connected serialized
graph. Component tests alone must not approve the original full solution.
