"""Shared license-policy parsing and matching for atoms and artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


def _split_csv(value: str) -> tuple[str, ...]:
    items: list[str] = []
    for part in str(value or "").split(","):
        token = part.strip()
        if token:
            items.append(token)
    return tuple(items)


def _normalize_expression(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_family(value: str) -> str:
    return str(value or "").strip().lower()


@dataclass(frozen=True)
class LicensePolicy:
    allowed_expressions: tuple[str, ...] = ()
    allowed_families: tuple[str, ...] = ()
    allow_unknown: bool = False
    enforce_status: bool = True

    @property
    def enabled(self) -> bool:
        return self.enforce_status or bool(self.allowed_expressions) or bool(self.allowed_families)

    def permits(
        self,
        *,
        license_expression: str,
        license_status: str,
        license_family: str,
    ) -> bool:
        status = str(license_status or "").strip().lower()
        expression = _normalize_expression(license_expression)
        family = _normalize_family(license_family)

        if self.enforce_status and status not in {"approved"}:
            return False

        if not expression or expression.upper() == "NOASSERTION":
            return self.allow_unknown

        if self.allowed_expressions and expression not in self.allowed_expressions:
            return False

        if self.allowed_families and family not in self.allowed_families:
            return False

        return True


def load_license_policy_from_env(
    *,
    developer_mode: bool = False,
    default_allow_unknown: bool | None = None,
    default_enforce_status: bool | None = None,
) -> LicensePolicy:
    expressions = _split_csv(os.environ.get("SCIONA_ALLOWED_LICENSES", ""))
    families = tuple(_normalize_family(item) for item in _split_csv(os.environ.get("SCIONA_ALLOWED_LICENSE_FAMILIES", "")))

    raw_allow_unknown = os.environ.get("SCIONA_ALLOW_UNKNOWN_LICENSES", "").strip().lower()
    if raw_allow_unknown:
        allow_unknown = raw_allow_unknown in {"1", "true", "yes", "on"}
    else:
        if default_allow_unknown is None:
            allow_unknown = developer_mode
        else:
            allow_unknown = default_allow_unknown

    raw_enforce_status = os.environ.get("SCIONA_ENFORCE_LICENSE_STATUS", "").strip().lower()
    if raw_enforce_status:
        enforce_status = raw_enforce_status in {"1", "true", "yes", "on"}
    else:
        if default_enforce_status is None:
            enforce_status = not developer_mode
        else:
            enforce_status = default_enforce_status

    return LicensePolicy(
        allowed_expressions=expressions,
        allowed_families=families,
        allow_unknown=allow_unknown,
        enforce_status=enforce_status,
    )


def summarize_license_rows(rows: Iterable[dict]) -> dict[str, int]:
    summary = {
        "approved": 0,
        "restricted": 0,
        "unknown": 0,
        "needs_legal_review": 0,
        "missing": 0,
    }
    for row in rows:
        status = str(row.get("license_status", "") or "").strip().lower()
        if status in summary:
            summary[status] += 1
        else:
            summary["missing"] += 1
    return summary
