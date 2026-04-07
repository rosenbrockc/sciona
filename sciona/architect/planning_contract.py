"""Phase 1 planning/constraint artifact for Architect decomposition runs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from sciona.architect.models import IOSpec


class PlanningConstraintCategory(str, Enum):
    """Canonical constraint buckets for Phase 1 planning."""

    DATA_KIND = "data_kind"
    PROVENANCE = "provenance"
    LOSS = "loss"
    STAGE = "stage"
    ADMISSIBILITY = "admissibility"
    TELEMETRY = "telemetry"


class PlanningConstraint(BaseModel):
    """One auditable planning constraint."""

    category: PlanningConstraintCategory
    subject: str = ""
    statement: str
    severity: Literal["required", "advisory"] = "required"
    rationale: str = ""
    source_stage: str = ""
    source_reference: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PlanningPortContract(BaseModel):
    """A first-class input or output contract declared during planning."""

    role: Literal["input", "output"]
    name: str
    type_desc: str
    data_kind: str = ""
    constraints: str = ""
    provenance: str = ""
    required: bool = True


class PlanningArtifact(BaseModel):
    """Canonical planning artifact emitted before the first candidate CDG."""

    artifact_version: str = "phase1.v1"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    goal: str
    thread_id: str
    paradigm: str = ""
    family_hint: str = ""
    strategy_rationale: str = ""
    skeleton_intent: dict[str, Any] = Field(default_factory=dict)
    input_contracts: list[PlanningPortContract] = Field(default_factory=list)
    output_contracts: list[PlanningPortContract] = Field(default_factory=list)
    planning_constraints: list[PlanningConstraint] = Field(default_factory=list)
    stage_constraints: dict[str, list[PlanningConstraint]] = Field(default_factory=dict)
    admissibility_expectations: list[str] = Field(default_factory=list)
    telemetry_expectations: list[str] = Field(default_factory=list)
    planning_assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


def infer_data_kind(type_desc: str, name: str = "") -> str:
    """Infer a canonical data-kind label from a contract type or port name."""
    text = f"{type_desc} {name}".lower()
    if any(token in text for token in ("waveform", "signal", "time series", "sample")):
        return "waveform"
    if any(token in text for token in ("event", "peak", "index", "spike", "beat")):
        return "event_sequence"
    if any(token in text for token in ("rate", "bpm", "frequency", "freq")):
        return "rate_series"
    if any(token in text for token in ("feature", "embedding", "vector")):
        return "feature_vector"
    if any(token in text for token in ("state", "hidden", "latent")):
        return "state"
    if "mask" in text:
        return "mask"
    if any(token in text for token in ("param", "hyperparam", "tunable")):
        return "parameter"
    if any(token in text for token in ("stat", "summary", "loss", "metric")):
        return "scalar_statistic"
    return "scalar_statistic" if "float" in text or "int" in text else "feature_vector"


def _build_port_contract(port: IOSpec, *, role: Literal["input", "output"]) -> PlanningPortContract:
    data_kind = infer_data_kind(port.type_desc, port.name)
    provenance = port.constraints.strip()
    return PlanningPortContract(
        role=role,
        name=port.name,
        type_desc=port.type_desc,
        data_kind=data_kind,
        constraints=port.constraints,
        provenance=provenance,
        required=port.required,
    )


def _port_constraints(
    port_contract: PlanningPortContract,
    *,
    stage: str,
) -> list[PlanningConstraint]:
    constraints = [
        PlanningConstraint(
            category=PlanningConstraintCategory.DATA_KIND,
            subject=f"{port_contract.role}:{port_contract.name}",
            statement=(
                f"{port_contract.role.title()} port '{port_contract.name}' "
                f"must preserve {port_contract.data_kind} semantics "
                f"({port_contract.type_desc})."
            ),
            source_stage=stage,
        )
    ]
    if port_contract.provenance:
        constraints.append(
            PlanningConstraint(
                category=PlanningConstraintCategory.PROVENANCE,
                subject=f"{port_contract.role}:{port_contract.name}",
                statement=(
                    f"Respect declared provenance / constraint: "
                    f"{port_contract.provenance}."
                ),
                source_stage=stage,
            )
        )
    return constraints


def build_planning_artifact(
    *,
    goal: str,
    thread_id: str,
    paradigm: str,
    family_hint: str = "",
    strategy_rationale: str = "",
    variant_hint: str = "",
    skeleton_instantiated: bool = False,
    root_inputs: list[IOSpec] | None = None,
    root_outputs: list[IOSpec] | None = None,
    skeleton_asset: dict[str, Any] | None = None,
    assumptions: list[str] | None = None,
    unresolved_questions: list[str] | None = None,
    extra_constraints: list[PlanningConstraint] | None = None,
) -> PlanningArtifact:
    """Construct the canonical planning artifact from strategy output."""
    inputs = [_build_port_contract(port, role="input") for port in root_inputs or []]
    outputs = [
        _build_port_contract(port, role="output") for port in root_outputs or []
    ]
    planning_constraints: list[PlanningConstraint] = []
    for contract in [*inputs, *outputs]:
        planning_constraints.extend(_port_constraints(contract, stage="strategy"))

    stage_constraints: dict[str, list[PlanningConstraint]] = {
        "strategy": [
            PlanningConstraint(
                category=PlanningConstraintCategory.STAGE,
                subject="planning",
                statement=(
                    "Identify a family-level plan before the first candidate CDG is "
                    "finalised."
                ),
                rationale="Phase 1 requires constraints to be explicit before decomposition.",
                source_stage="strategy",
            ),
            PlanningConstraint(
                category=PlanningConstraintCategory.LOSS,
                subject="planning",
                statement=(
                    "Avoid irreversible information loss before the stage that is "
                    "responsible for converting the relevant data kind."
                ),
                source_stage="strategy",
            ),
        ]
    }

    planning_assumptions = list(assumptions or [])
    if paradigm:
        planning_assumptions.append(f"Selected paradigm: {paradigm}")
    if family_hint and family_hint != paradigm:
        planning_assumptions.append(f"Family hint: {family_hint}")
    if variant_hint:
        planning_assumptions.append(f"Variant hint: {variant_hint}")
    if skeleton_instantiated:
        planning_assumptions.append("Skeleton template was instantiated for this run.")

    admissibility_expectations = [
        "Reject candidates that ignore declared input provenance or timing context.",
        "Reject candidates that introduce irreversible loss before the intended stage boundary.",
        "Reject candidates that cannot plausibly satisfy the declared output contract.",
    ]
    if any(contract.data_kind == "event_sequence" for contract in outputs):
        admissibility_expectations.append(
            "Reject candidates that collapse event count below a plausible extraction threshold."
        )
    telemetry_expectations = [
        "Track intermediate port summaries for each declared boundary transition.",
        "Record enough evidence to explain major data-kind changes during synthesis.",
        "Persist runtime summaries that can validate the declared output contract.",
    ]
    if any(contract.data_kind == "rate_series" for contract in outputs):
        telemetry_expectations.append(
            "Record rate-estimation support metrics such as event counts and interval stability."
        )

    artifact = PlanningArtifact(
        goal=goal,
        thread_id=thread_id,
        paradigm=paradigm,
        family_hint=family_hint or paradigm,
        strategy_rationale=strategy_rationale,
        skeleton_intent={
            "variant_hint": variant_hint,
            "skeleton_instantiated": skeleton_instantiated,
            "root_input_count": len(inputs),
            "root_output_count": len(outputs),
            "root_input_names": [contract.name for contract in inputs],
            "root_output_names": [contract.name for contract in outputs],
            "asset": dict(skeleton_asset or {}),
        },
        input_contracts=inputs,
        output_contracts=outputs,
        planning_constraints=[*planning_constraints, *(extra_constraints or [])],
        stage_constraints=stage_constraints,
        admissibility_expectations=admissibility_expectations,
        telemetry_expectations=telemetry_expectations,
        planning_assumptions=planning_assumptions,
        unresolved_questions=list(unresolved_questions or []),
    )
    return artifact


def summarize_planning_artifact(artifact: dict[str, Any] | PlanningArtifact | None) -> dict[str, Any]:
    """Return a compact, telemetry-friendly summary of the planning artifact."""
    if artifact is None:
        return {}
    if isinstance(artifact, PlanningArtifact):
        data = artifact.model_dump(mode="json")
    else:
        data = dict(artifact)
    constraints = data.get("planning_constraints", []) or []
    categories = Counter()
    for item in constraints:
        category = item.get("category", "") if isinstance(item, dict) else ""
        if category:
            categories[str(category)] += 1
    skeleton_intent = data.get("skeleton_intent", {}) if isinstance(data, dict) else {}
    skeleton_asset = skeleton_intent.get("asset", {}) if isinstance(skeleton_intent, dict) else {}
    return {
        "artifact_version": data.get("artifact_version", ""),
        "paradigm": data.get("paradigm", ""),
        "family_hint": data.get("family_hint", ""),
        "goal": data.get("goal", ""),
        "constraint_count": len(constraints),
        "constraint_categories": dict(categories),
        "input_contract_count": len(data.get("input_contracts", []) or []),
        "output_contract_count": len(data.get("output_contracts", []) or []),
        "admissibility_count": len(data.get("admissibility_expectations", []) or []),
        "telemetry_count": len(data.get("telemetry_expectations", []) or []),
        "assumption_count": len(data.get("planning_assumptions", []) or []),
        "unresolved_question_count": len(data.get("unresolved_questions", []) or []),
        "skeleton_variant_hint": str(skeleton_intent.get("variant_hint", "")),
        "skeleton_instantiated": bool(skeleton_intent.get("skeleton_instantiated", False)),
        "skeleton_asset_id": str(skeleton_asset.get("asset_id", "")),
        "skeleton_asset_version": str(skeleton_asset.get("asset_version", "")),
        "skeleton_asset_source_kind": str(skeleton_asset.get("source_kind", "")),
    }


def render_planning_artifact_block(artifact: dict[str, Any] | PlanningArtifact | None) -> str:
    """Render the artifact as a compact prompt block."""
    if artifact is None:
        return ""
    if isinstance(artifact, PlanningArtifact):
        data = artifact.model_dump(mode="json")
    else:
        data = dict(artifact)
    if not data:
        return ""

    lines: list[str] = ["Planning context:"]
    lines.append(
        f"  - paradigm: {data.get('paradigm', '') or '(unknown)'}"
        f" | family_hint: {data.get('family_hint', '') or '(unknown)'}"
    )
    rationale = str(data.get("strategy_rationale", "")).strip()
    if rationale:
        lines.append(f"  - strategy_rationale: {rationale}")

    skeleton_intent = data.get("skeleton_intent", {}) or {}
    if skeleton_intent:
        lines.append("  - skeleton_intent:")
        for key in (
            "variant_hint",
            "skeleton_instantiated",
            "root_input_count",
            "root_output_count",
        ):
            if key in skeleton_intent:
                lines.append(f"      {key}: {skeleton_intent.get(key)}")
        asset = skeleton_intent.get("asset", {})
        if isinstance(asset, dict) and asset:
            lines.append("      asset:")
            for key in ("asset_id", "asset_version", "source_kind", "family"):
                if key in asset:
                    lines.append(f"        {key}: {asset.get(key)}")

    def _render_contracts(title: str, contracts: list[dict[str, Any]]) -> None:
        if not contracts:
            return
        lines.append(f"  - {title}:")
        for contract in contracts[:8]:
            lines.append(
                "      "
                f"{contract.get('role', 'port')}:{contract.get('name', '')} "
                f"[{contract.get('data_kind', '')}] {contract.get('type_desc', '')}"
                + (
                    f" | constraints: {contract.get('constraints', '')}"
                    if contract.get("constraints")
                    else ""
                )
            )

    _render_contracts("input_contracts", data.get("input_contracts", []) or [])
    _render_contracts("output_contracts", data.get("output_contracts", []) or [])

    def _render_list(title: str, values: list[Any]) -> None:
        if not values:
            return
        lines.append(f"  - {title}:")
        for value in values[:8]:
            lines.append(f"      - {value}")

    _render_list("planning_constraints", [
        f"[{item.get('category', '')}] {item.get('subject', '')}: {item.get('statement', '')}"
        for item in data.get("planning_constraints", []) or []
        if isinstance(item, dict)
    ])
    _render_list("admissibility_expectations", data.get("admissibility_expectations", []) or [])
    _render_list("telemetry_expectations", data.get("telemetry_expectations", []) or [])
    _render_list("planning_assumptions", data.get("planning_assumptions", []) or [])
    _render_list("unresolved_questions", data.get("unresolved_questions", []) or [])
    return "\n".join(lines)
