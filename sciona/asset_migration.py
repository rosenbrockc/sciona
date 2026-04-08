"""Shared migration-readiness contracts for auditable family assets."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MigrationReadinessCheck(BaseModel):
    """One auditable checklist item for asset migration readiness."""

    check_id: str
    description: str
    required: bool = True
    satisfied: bool = False
    evidence: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class MigrationReadinessAsset(BaseModel):
    """Cross-family readiness contract for promoting assets into shared ownership."""

    status: Literal[
        "not_assessed",
        "in_progress",
        "ready_for_migration",
        "migrated",
    ] = "not_assessed"
    target_repository: str = "../ageo-atoms"
    target_scope: str = ""
    rationale: str = ""
    checklist: list[MigrationReadinessCheck] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    reviewers: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.status in {"ready_for_migration", "migrated"}:
            missing = [
                item.check_id
                for item in self.checklist
                if item.required and not item.satisfied
            ]
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    "Migration-ready assets must satisfy every required checklist item: "
                    f"{missing_str}"
                )

    def required_check_count(self) -> int:
        return sum(1 for item in self.checklist if item.required)

    def completed_required_check_count(self) -> int:
        return sum(1 for item in self.checklist if item.required and item.satisfied)

    def is_ready_for_migration(self) -> bool:
        return self.status in {"ready_for_migration", "migrated"} and (
            self.required_check_count() == self.completed_required_check_count()
        )


def migration_readiness_summary(
    readiness: MigrationReadinessAsset | dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a compact runtime summary for migration-readiness metadata."""
    if readiness is None:
        return {}
    if isinstance(readiness, MigrationReadinessAsset):
        data = readiness.model_dump(mode="json")
        required = readiness.required_check_count()
        completed = readiness.completed_required_check_count()
        ready = readiness.is_ready_for_migration()
    elif isinstance(readiness, dict):
        data = dict(readiness)
        checklist = data.get("checklist", [])
        if not isinstance(checklist, list):
            checklist = []
        required = sum(
            1
            for item in checklist
            if isinstance(item, dict) and bool(item.get("required", True))
        )
        completed = sum(
            1
            for item in checklist
            if isinstance(item, dict)
            and bool(item.get("required", True))
            and bool(item.get("satisfied"))
        )
        status = str(data.get("status", "") or "")
        ready = status in {"ready_for_migration", "migrated"} and (
            required == completed
        )
    else:
        return {}

    checklist = data.get("checklist", [])
    if not isinstance(checklist, list):
        checklist = []
    return {
        "migration_readiness_status": str(data.get("status", "") or ""),
        "migration_readiness_target_repository": str(
            data.get("target_repository", "") or ""
        ),
        "migration_readiness_target_scope": str(data.get("target_scope", "") or ""),
        "migration_readiness_rationale": str(data.get("rationale", "") or ""),
        "migration_readiness_check_count": len(checklist),
        "migration_readiness_required_check_count": required,
        "migration_readiness_completed_required_check_count": completed,
        "migration_readiness_ready": ready,
        "migration_readiness_check_ids": [
            str(item.get("check_id", ""))
            for item in checklist
            if isinstance(item, dict) and item.get("check_id")
        ],
    }
