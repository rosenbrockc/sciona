"""Hyperparameter manifest loader and built-in runtime param definitions."""

from __future__ import annotations

import json
import logging
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

from sciona.architect.models import PrimitiveParamSpec

logger = logging.getLogger(__name__)

_MANIFEST_MAX_AGE_DAYS = 30


def _infer_kind(value: object, has_choices: bool) -> str:
    """Infer PrimitiveParamSpec.kind from a Python value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if has_choices or isinstance(value, str):
        return "categorical"
    return "float"


def _decode_json_text(text: str | None) -> object:
    """Decode a JSON-encoded TEXT column, returning None for SQL NULL or ``"null"``."""
    if text is None:
        return None
    return json.loads(text)


def _normalize_manifest_param(raw: dict[str, object]) -> dict[str, object]:
    """Translate provider manifest JSON keys into ``PrimitiveParamSpec`` keys."""
    data = dict(raw)
    if "min_value" not in data and "min" in data:
        data["min_value"] = data.pop("min")
    if "max_value" not in data and "max" in data:
        data["max_value"] = data.pop("max")
    constraints = data.get("constraints")
    if isinstance(constraints, dict):
        data["constraints"] = json.dumps(constraints, sort_keys=True)
    elif constraints is None:
        data["constraints"] = ""
    else:
        data["constraints"] = str(constraints)
    return data


def _read_manifest_metadata(con: sqlite3.Connection) -> dict[str, str]:
    tables = {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "manifest_metadata" not in tables:
        return {}

    columns = {
        row[1]
        for row in con.execute("PRAGMA table_info(manifest_metadata)").fetchall()
    }
    if {"key", "value"}.issubset(columns):
        return {
            str(row["key"]): str(row["value"])
            for row in con.execute("SELECT key, value FROM manifest_metadata").fetchall()
        }

    row = con.execute("SELECT * FROM manifest_metadata LIMIT 1").fetchone()
    if row is None:
        return {}
    return {key: str(row[key]) for key in row.keys() if row[key] is not None}


def _manifest_atoms_content_hash(con: sqlite3.Connection) -> str | None:
    try:
        rows = con.execute("SELECT fqdn FROM atoms ORDER BY fqdn ASC").fetchall()
    except sqlite3.Error:
        return None
    import hashlib

    payload = "\n".join(str(row["fqdn"]) for row in rows if row["fqdn"]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_manifest_sqlite_health(
    con: sqlite3.Connection,
    db_path: Path,
    *,
    max_age_days: int = _MANIFEST_MAX_AGE_DAYS,
) -> None:
    metadata = _read_manifest_metadata(con)
    if not metadata:
        return

    generated_at = metadata.get("generated_at", "").strip()
    if generated_at:
        normalized = generated_at.replace("Z", "+00:00")
        try:
            generated = datetime.fromisoformat(normalized)
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - generated.astimezone(timezone.utc)
            if age.days > max_age_days:
                warnings.warn(
                    (
                        f"{db_path} is {age.days} days old. "
                        "Run 'sciona catalog sync' to update."
                    ),
                    stacklevel=2,
                )
        except ValueError:
            pass

    expected_hash = metadata.get("content_hash", "").strip()
    if expected_hash:
        actual_hash = _manifest_atoms_content_hash(con)
        if actual_hash and actual_hash != expected_hash:
            warnings.warn(
                f"{db_path} failed manifest content-hash validation.",
                stacklevel=2,
            )


def load_hyperparams_manifest_sqlite(
    db_path: Path,
) -> dict[str, list[PrimitiveParamSpec]]:
    """Load tunables from a provider manifest.sqlite.

    Queries only approved atoms with approved hyperparams.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("Hyperparams SQLite manifest not found: %s", db_path)
        return {}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        _check_manifest_sqlite_health(con, db_path)
        rows = con.execute(
            """
            SELECT a.fqdn, h.name, h.default_value, h.min_value, h.max_value,
                   h.step_value, h.log_scale, h.choices_json, h.constraints_json,
                   h.semantic_role
            FROM atoms a
            JOIN hyperparams h ON a.atom_id = h.atom_id
            WHERE a.status = 'approved'
              AND h.status = 'approved'
            """
        ).fetchall()
    finally:
        con.close()

    result: dict[str, list[PrimitiveParamSpec]] = {}
    for row in rows:
        fqdn = row["fqdn"]
        default = _decode_json_text(row["default_value"])
        choices_raw = _decode_json_text(row["choices_json"])
        choices = list(choices_raw) if isinstance(choices_raw, list) else None
        constraints_raw = _decode_json_text(row["constraints_json"])
        constraints = str(constraints_raw) if constraints_raw else ""

        kind = _infer_kind(default, choices is not None and len(choices) > 0)

        try:
            spec = PrimitiveParamSpec(
                name=row["name"],
                kind=kind,
                default=default if default is not None else 0,
                min_value=_decode_json_text(row["min_value"]),
                max_value=_decode_json_text(row["max_value"]),
                step=_decode_json_text(row["step_value"]),
                log_scale=bool(row["log_scale"]),
                choices=choices,
                constraints=constraints,
                semantic_role=row["semantic_role"] or "",
                safe_to_optimize=True,
            )
        except Exception:
            logger.warning("Skipping invalid param %s on atom %s", row["name"], fqdn)
            continue

        result.setdefault(fqdn, []).append(spec)

    return result


def load_hyperparams_manifest(
    manifest_path: Path,
) -> dict[str, list[PrimitiveParamSpec]]:
    """Load a provider manifest.json and return atom_name -> tunable_params mapping.

    Only returns params from atoms with status="approved" and safe_to_optimize=True.
    Falls back to this when SQLite manifest is unavailable.
    """
    path = Path(manifest_path)
    if not path.exists():
        logger.warning("Hyperparams manifest not found: %s", path)
        return {}

    raw = json.loads(path.read_text())
    atoms: list[dict] = raw.get("reviewed_atoms", [])
    result: dict[str, list[PrimitiveParamSpec]] = {}

    for atom in atoms:
        status = atom.get("status", "")
        if status != "approved":
            continue
        atom_name = atom.get("atom", "")
        if not atom_name:
            continue

        params: list[PrimitiveParamSpec] = []
        for p in atom.get("tunable_params", []):
            p = _normalize_manifest_param(p)
            # Infer kind if not explicitly provided
            if "kind" not in p and "default" in p:
                has_choices = bool(p.get("choices"))
                p = {**p, "kind": _infer_kind(p["default"], has_choices)}
            try:
                spec = PrimitiveParamSpec(**p)
            except Exception:
                logger.warning(
                    "Skipping invalid param %s on atom %s",
                    p.get("name", "?"),
                    atom_name,
                )
                continue
            if spec.safe_to_optimize:
                params.append(spec)

        if params:
            result[atom_name] = params

    return result


def load_manifest(manifest_path: Path) -> dict[str, list[PrimitiveParamSpec]]:
    """Load tunables from manifest. Prefers .sqlite, falls back to .json."""
    sqlite_path = Path(manifest_path).with_suffix(".sqlite")
    if sqlite_path.exists():
        try:
            return load_hyperparams_manifest_sqlite(sqlite_path)
        except sqlite3.Error as exc:
            logger.warning(
                "Falling back to JSON manifest after SQLite load failure: %s",
                exc,
            )
    if Path(manifest_path).exists():
        return load_hyperparams_manifest(manifest_path)
    return {}


def get_runtime_signal_event_rate_params() -> dict[str, list[PrimitiveParamSpec]]:
    """Return hand-audited tunable params for the built-in signal_event_rate functions."""
    return {
        "filter_signal_for_detection": [
            PrimitiveParamSpec(
                name="filter_order",
                kind="int",
                default=4,
                min_value=2,
                max_value=8,
                step=2,
                semantic_role="Butterworth filter order",
                range_source="signal processing convention",
                source_confidence="high",
            ),
            PrimitiveParamSpec(
                name="clipping_scale",
                kind="float",
                default=8.0,
                min_value=3.0,
                max_value=15.0,
                semantic_role="Outlier clipping threshold in MAD units",
                range_source="empirical",
                source_confidence="medium",
            ),
            PrimitiveParamSpec(
                name="low_cutoff_hz",
                kind="float",
                default=3.0,
                min_value=0.5,
                max_value=10.0,
                semantic_role="Bandpass low cutoff frequency",
                range_source="physiological signal range",
                source_confidence="high",
            ),
            PrimitiveParamSpec(
                name="high_cutoff_hz",
                kind="float",
                default=25.0,
                min_value=10.0,
                max_value=50.0,
                semantic_role="Bandpass high cutoff frequency",
                range_source="physiological signal range",
                source_confidence="high",
            ),
        ],
        "detect_peaks_in_signal": [
            PrimitiveParamSpec(
                name="prominence_scale",
                kind="float",
                default=1.5,
                min_value=0.5,
                max_value=5.0,
                semantic_role="Peak prominence threshold in MAD units",
                range_source="empirical",
                source_confidence="medium",
            ),
            PrimitiveParamSpec(
                name="refractory_scale",
                kind="float",
                default=0.45,
                min_value=0.2,
                max_value=0.8,
                semantic_role="Refractory period as fraction of sampling rate",
                range_source="physiological minimum IBI",
                source_confidence="high",
            ),
        ],
        "compute_event_rate_smoothed": [
            PrimitiveParamSpec(
                name="smoothing_window",
                kind="int",
                default=5,
                min_value=1,
                max_value=15,
                step=2,
                semantic_role="Moving average window size for rate smoothing",
                range_source="empirical",
                source_confidence="medium",
            ),
        ],
        "compute_event_rate_median_smoothed": [
            PrimitiveParamSpec(
                name="smoothing_window",
                kind="int",
                default=5,
                min_value=1,
                max_value=15,
                step=2,
                semantic_role="Median smoothing window size for robust rate aggregation",
                range_source="empirical",
                source_confidence="medium",
            ),
        ],
    }
