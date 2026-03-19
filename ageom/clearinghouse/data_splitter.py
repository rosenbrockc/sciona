"""Deterministic blind data splitting pipeline.

Splits dataset into 20% public / 80% blind partitions using a
bounty-specific salt to prevent cross-bounty replay attacks.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from ageom.clearinghouse.models import SplitAssignment, SplitResult


def compute_split_hash(assignments: Sequence[SplitAssignment], bounty_salt: str) -> str:
    """Compute the canonical split hash.

    The hash binds the split to a specific bounty via the salt.
    """
    canonical = json.dumps(
        [{"key": a.unit_key, "partition": a.partition} for a in sorted(assignments, key=lambda x: x.unit_key)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256((canonical + bounty_salt).encode("utf-8")).hexdigest()


def assign_partition(unit_key: str, bounty_id: str) -> str:
    """Deterministically assign a data unit to public or blind partition.

    Uses ``SHA-256(bounty_id:unit_key) mod 5 == 0`` for ~20% public / ~80% blind.
    """
    digest = hashlib.sha256(f"{bounty_id}:{unit_key}".encode("utf-8")).hexdigest()
    return "public" if int(digest, 16) % 5 == 0 else "blind"


def split_dataset(
    unit_keys: Sequence[str],
    bounty_id: str,
    *,
    min_samples: int = 50,
    min_public_samples: int = 10,
) -> SplitResult:
    """Split a dataset into public/blind partitions.

    Parameters
    ----------
    unit_keys
        Unique identifiers for each data unit (e.g., subject ID, file hash).
    bounty_id
        The bounty ID, used as salt for deterministic assignment.
    min_samples
        Minimum total samples required.
    min_public_samples
        Minimum samples required in the public partition.

    Raises
    ------
    ValueError
        If the dataset is too small or the split is degenerate.
    """
    if len(unit_keys) < min_samples:
        raise ValueError(
            f"Dataset has {len(unit_keys)} samples, minimum is {min_samples}"
        )

    assignments = [
        SplitAssignment(unit_key=key, partition=assign_partition(key, bounty_id))
        for key in unit_keys
    ]

    public_count = sum(1 for a in assignments if a.partition == "public")
    blind_count = len(assignments) - public_count

    if public_count < min_public_samples:
        raise ValueError(
            f"Public split has {public_count} samples, minimum is {min_public_samples}"
        )

    split_hash = compute_split_hash(assignments, bounty_id)

    return SplitResult(
        bounty_id=bounty_id,
        split_hash=split_hash,
        public_count=public_count,
        blind_count=blind_count,
        assignments=assignments,
    )


def validate_dataset(
    data_stats: dict[str, Any],
    *,
    min_samples: int = 50,
) -> list[str]:
    """Validate dataset statistics before splitting.

    Returns a list of rejection reasons (empty if valid).
    """
    reasons: list[str] = []

    total = data_stats.get("total_samples", 0)
    if total < min_samples:
        reasons.append(f"Too few samples: {total} < {min_samples}")

    constant_columns = data_stats.get("constant_columns", [])
    if constant_columns:
        reasons.append(f"All-constant columns: {constant_columns}")

    return reasons
