"""SQLite snapshot generation — bridge between platform PostgreSQL and offline CLI."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def generate_manifest_sqlite(
    atoms: list[dict[str, Any]],
    hyperparams: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
) -> sqlite3.Connection:
    """Generate a manifest.sqlite from platform data.

    Parameters
    ----------
    atoms
        List of dicts with keys: ``atom_id``, ``fqdn``, ``status``,
        ``domain_tags``, ``description``.
    hyperparams
        List of dicts matching the ``hyperparams`` table schema:
        ``hp_id``, ``atom_id``, ``name``, ``kind``, ``default_value``,
        ``min_value``, ``max_value``, ``step_value``, ``log_scale``,
        ``choices_json``, ``constraints_json``, ``semantic_role``, ``status``.
    benchmarks
        Optional list of benchmark records: ``atom_fqdn``, ``content_hash``,
        ``benchmark_name``, ``metric_name``, ``metric_value``, ``dataset_tag``,
        ``measured_at``.
    output_path
        If provided, write the database to this file. Otherwise return
        an in-memory connection.

    Returns
    -------
    sqlite3.Connection
        The SQLite database (in-memory or on-disk).
    """
    db_str = str(output_path) if output_path else ":memory:"
    con = sqlite3.connect(db_str)

    con.execute(
        """CREATE TABLE IF NOT EXISTS atoms (
            atom_id   TEXT PRIMARY KEY,
            fqdn      TEXT UNIQUE NOT NULL,
            status    TEXT NOT NULL DEFAULT 'approved',
            domain_tags TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT ''
        )"""
    )

    con.execute(
        """CREATE TABLE IF NOT EXISTS hyperparams (
            hp_id             TEXT PRIMARY KEY,
            atom_id           TEXT NOT NULL,
            name              TEXT NOT NULL,
            kind              TEXT NOT NULL,
            default_value     TEXT,
            min_value         TEXT,
            max_value         TEXT,
            step_value        TEXT,
            log_scale         INTEGER NOT NULL DEFAULT 0,
            choices_json      TEXT,
            constraints_json  TEXT,
            semantic_role     TEXT NOT NULL DEFAULT '',
            status            TEXT NOT NULL DEFAULT 'approved',
            UNIQUE (atom_id, name)
        )"""
    )

    con.execute(
        """CREATE TABLE IF NOT EXISTS benchmarks (
            atom_fqdn       TEXT NOT NULL,
            content_hash    TEXT NOT NULL,
            benchmark_name  TEXT NOT NULL,
            metric_name     TEXT NOT NULL,
            metric_value    REAL NOT NULL,
            dataset_tag     TEXT NOT NULL DEFAULT '',
            measured_at     TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (atom_fqdn, content_hash, benchmark_name, metric_name)
        )"""
    )

    for atom in atoms:
        tags = atom.get("domain_tags", [])
        tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)
        con.execute(
            "INSERT OR REPLACE INTO atoms (atom_id, fqdn, status, domain_tags, description) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(atom["atom_id"]),
                atom["fqdn"],
                atom.get("status", "approved"),
                tags_str,
                atom.get("description", ""),
            ),
        )

    for hp in hyperparams:
        con.execute(
            """INSERT OR REPLACE INTO hyperparams
               (hp_id, atom_id, name, kind, default_value, min_value, max_value,
                step_value, log_scale, choices_json, constraints_json, semantic_role, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(hp.get("hp_id", "")),
                str(hp["atom_id"]),
                hp["name"],
                hp["kind"],
                str(hp.get("default_value")) if hp.get("default_value") is not None else None,
                str(hp.get("min_value")) if hp.get("min_value") is not None else None,
                str(hp.get("max_value")) if hp.get("max_value") is not None else None,
                str(hp.get("step_value")) if hp.get("step_value") is not None else None,
                int(hp.get("log_scale", False)),
                str(hp.get("choices_json")) if hp.get("choices_json") is not None else None,
                str(hp.get("constraints_json")) if hp.get("constraints_json") is not None else None,
                hp.get("semantic_role", ""),
                hp.get("status", "approved"),
            ),
        )

    for bm in benchmarks or []:
        con.execute(
            """INSERT OR REPLACE INTO benchmarks
               (atom_fqdn, content_hash, benchmark_name, metric_name,
                metric_value, dataset_tag, measured_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                bm["atom_fqdn"],
                bm["content_hash"],
                bm["benchmark_name"],
                bm["metric_name"],
                bm["metric_value"],
                bm.get("dataset_tag", ""),
                bm.get("measured_at", ""),
            ),
        )

    con.commit()
    return con
