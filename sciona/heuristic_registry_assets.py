"""Reference heuristic registry assets for family-local interpretation."""

from __future__ import annotations

from sciona.heuristic_registries import (
    HeuristicActionClass,
    HeuristicFamilyRegistry,
    HeuristicRegistryAudit,
    HeuristicRegistryEntry,
    HeuristicRegistryReference,
)
from sciona.heuristics import HeuristicProducerKind


def _signal_processing_registry() -> HeuristicFamilyRegistry:
    return HeuristicFamilyRegistry(
        asset_id="registry.signal_processing.heuristics.v1",
        asset_version="phase3.v1",
        family="signal_processing",
        skeleton_scope="family",
        name="Signal Processing Heuristic Registry",
        summary=(
            "Family-local interpretation of canonical heuristics for sampled-signal "
            "pipelines."
        ),
        dejargonized_summary=(
            "These are family-local notes for when signal pipelines should clean up, "
            "gate, or reshape the graph."
        ),
        entries=[
            HeuristicRegistryEntry(
                heuristic_id="boundary_discontinuity",
                sanctioned_producer_kinds=[HeuristicProducerKind.RUNTIME_TRANSFORM],
                supported_action_classes=[
                    HeuristicActionClass.PRECONDITION,
                    HeuristicActionClass.INSERT_CORRECTION,
                ],
                action_priority=[
                    HeuristicActionClass.PRECONDITION,
                    HeuristicActionClass.INSERT_CORRECTION,
                ],
                admissibility_notes=[
                    "Use when a waveform boundary contains abrupt discontinuities that can confuse downstream conditioning."
                ],
                escalation_conditions=[
                    "Escalate when the boundary path remains available before the first conditioning stage."
                ],
                family_notes=[
                    "In signal processing this often means a large waveform step or jump before filtering.",
                    "Signal-processing meaning remains local; the shared heuristic id stays boundary_discontinuity.",
                ],
            ),
            HeuristicRegistryEntry(
                heuristic_id="quality_instability",
                sanctioned_producer_kinds=[HeuristicProducerKind.RUNTIME_TRANSFORM],
                supported_action_classes=[
                    HeuristicActionClass.GATE_OR_VALIDATE,
                    HeuristicActionClass.BRANCH_AND_COMPARE,
                ],
                action_priority=[
                    HeuristicActionClass.GATE_OR_VALIDATE,
                    HeuristicActionClass.BRANCH_AND_COMPARE,
                ],
                admissibility_notes=[
                    "Use when a waveform quality score varies enough to make one conditioning path brittle."
                ],
                escalation_conditions=[
                    "Escalate when a quality gate can be inserted without hiding the waveform path."
                ],
                family_notes=[
                    "This is the family-local quality view that may motivate an explicit quality gate.",
                    "This registry does not redefine the heuristic id.",
                ],
            ),
            HeuristicRegistryEntry(
                heuristic_id="interval_instability",
                sanctioned_producer_kinds=[HeuristicProducerKind.RUNTIME_TRANSFORM],
                supported_action_classes=[
                    HeuristicActionClass.SMOOTH_OR_AGGREGATE,
                    HeuristicActionClass.INSERT_CORRECTION,
                ],
                action_priority=[
                    HeuristicActionClass.SMOOTH_OR_AGGREGATE,
                    HeuristicActionClass.INSERT_CORRECTION,
                ],
                admissibility_notes=[
                    "Use when event intervals are internally inconsistent and a smoothing or rejection response is likely to help."
                ],
                escalation_conditions=[
                    "Escalate when the event stream remains available for post-detection refinement."
                ],
                family_notes=[
                    "This is the family view that supports event rejection or smoothing before rate estimation.",
                    "The meaning is local to the sampled-signal family.",
                ],
            ),
        ],
        audit=HeuristicRegistryAudit(
            source_kind="local_asset",
            review_status="transitional",
            rationale="Family-local interpretation of canonical heuristics for signal-processing pipelines.",
            dejargonized_summary=(
                "These are the family-specific notes for when signal pipelines should "
                "treat canonical heuristic signals as reasons to clean up, gate, or "
                "reshape the graph."
            ),
            references=[
                HeuristicRegistryReference(
                    title="Heuristic Evidence Layer Plan",
                    note="Ground-truth cross-family heuristic abstraction plan.",
                )
            ],
            maintainers=["ageo-matcher"],
        ),
    )


def _divide_and_conquer_registry() -> HeuristicFamilyRegistry:
    return HeuristicFamilyRegistry(
        asset_id="registry.divide_and_conquer.heuristics.v1",
        asset_version="phase3.v1",
        family="divide_and_conquer",
        skeleton_scope="family",
        name="Divide And Conquer Heuristic Registry",
        summary=(
            "Family-local interpretation of canonical heuristics for recursive "
            "split/merge style pipelines."
        ),
        dejargonized_summary=(
            "These are family-specific notes for when recursive pipelines should "
            "prune or branch differently because the current split is too thin."
        ),
        entries=[
            HeuristicRegistryEntry(
                heuristic_id="density_collapse",
                sanctioned_producer_kinds=[HeuristicProducerKind.RUNTIME_TRANSFORM],
                supported_action_classes=[
                    HeuristicActionClass.BRANCH_AND_COMPARE,
                    HeuristicActionClass.GATE_OR_VALIDATE,
                ],
                action_priority=[
                    HeuristicActionClass.BRANCH_AND_COMPARE,
                    HeuristicActionClass.GATE_OR_VALIDATE,
                ],
                admissibility_notes=[
                    "Use when recursive partitions become too uneven or too sparse to support reliable downstream merging."
                ],
                escalation_conditions=[
                    "Escalate when a subproblem branch appears too small or too imbalanced."
                ],
                family_notes=[
                    "In divide-and-conquer terms this is about a branch becoming too thin to justify the current split.",
                    "This registry treats density as branch coverage, not signal amplitude."
                ],
            ),
            HeuristicRegistryEntry(
                heuristic_id="constraint_violation_risk",
                sanctioned_producer_kinds=[HeuristicProducerKind.RUNTIME_TRANSFORM],
                supported_action_classes=[
                    HeuristicActionClass.PRECONDITION,
                    HeuristicActionClass.GATE_OR_VALIDATE,
                ],
                action_priority=[
                    HeuristicActionClass.PRECONDITION,
                    HeuristicActionClass.GATE_OR_VALIDATE,
                ],
                admissibility_notes=[
                    "Use when recursive choices are likely to violate declared feasibility or planning constraints."
                ],
                escalation_conditions=[
                    "Escalate when pruning can remove the risk before recursion deepens."
                ],
                family_notes=[
                    "This family interprets the heuristic as a pruning or feasibility issue.",
                    "This keeps the shared heuristic name unchanged.",
                ],
            ),
        ],
        audit=HeuristicRegistryAudit(
            source_kind="local_asset",
            review_status="draft",
            rationale="Non-signal reference registry to prove the interface stays generic.",
            dejargonized_summary=(
                "These are the family-specific notes for when recursive pipelines "
                "should prune or branch differently because the current split is too thin."
            ),
            references=[
                HeuristicRegistryReference(
                    title="Heuristic Evidence Layer Plan",
                    note="Ground-truth heuristic abstraction plan.",
                )
            ],
            maintainers=["ageo-matcher"],
        ),
    )


REFERENCE_HEURISTIC_REGISTRIES: tuple[HeuristicFamilyRegistry, ...] = (
    _signal_processing_registry(),
    _divide_and_conquer_registry(),
)
