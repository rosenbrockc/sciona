"""Deterministic ingest-time smoke validation.

This module intentionally keeps probe coverage narrow. The goal is to catch
obviously bad generated outputs for a small allowlisted subset, not to replay
the full audit stack from ``ageo-atoms`` inside the matcher.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SMOKE_STATUS_PASS = "pass"
SMOKE_STATUS_FAIL = "fail"
SMOKE_STATUS_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class SmokeResult:
    status: str
    target_symbol: str
    probe_id: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target_symbol": self.target_symbol,
            "probe_id": self.probe_id,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SmokeProbe:
    probe_id: str
    target_symbol: str
    runner: Callable[[Callable[..., Any]], None]


def _run_safe_grouped_atom_probe(fn: Callable[..., Any]) -> None:
    positive = fn(3)
    if positive != 4:
        raise AssertionError(f"expected safe_grouped_atom(3) == 4, got {positive!r}")
    try:
        fn("bad")
    except TypeError:
        return
    raise AssertionError("expected safe_grouped_atom('bad') to raise TypeError")


ALLOWLISTED_SMOKE_PROBES: dict[str, SmokeProbe] = {
    "safe_grouped_atom": SmokeProbe(
        probe_id="safe_grouped_atom.basic_increment",
        target_symbol="safe_grouped_atom",
        runner=_run_safe_grouped_atom_probe,
    ),
}


@contextmanager
def _module_import_path(path: Path):
    path_str = str(path)
    original = list(sys.path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        sys.path[:] = original


def _clear_module(module_name: str) -> None:
    doomed = [
        name
        for name in sys.modules
        if name == module_name or name.startswith(module_name + ".")
    ]
    for name in doomed:
        sys.modules.pop(name, None)


def _import_atoms_module(output_dir: Path):
    package_name = output_dir.name
    module_name = f"{package_name}.atoms"
    _clear_module(package_name)
    with _module_import_path(output_dir.parent):
        return importlib.import_module(module_name)


def run_smoke_validation(
    staged_dir: str | Path,
    *,
    package_basename: str,
    target_symbol: str,
) -> dict[str, Any]:
    staged_path = Path(staged_dir)
    probe = ALLOWLISTED_SMOKE_PROBES.get(target_symbol)
    if probe is None:
        return SmokeResult(
            status=SMOKE_STATUS_NOT_APPLICABLE,
            target_symbol=target_symbol,
            probe_id="",
            message="no allowlisted smoke probe for target",
            details={},
        ).to_dict()

    try:
        with tempfile.TemporaryDirectory(prefix="sciona_ingest_smoke_") as tmp_root:
            package_dir = Path(tmp_root) / package_basename
            package_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(staged_path.iterdir()):
                if path.is_file():
                    shutil.copy2(path, package_dir / path.name)
            module = _import_atoms_module(package_dir)
    except Exception as exc:
        return SmokeResult(
            status=SMOKE_STATUS_FAIL,
            target_symbol=target_symbol,
            probe_id=probe.probe_id,
            message="failed to import generated atoms module",
            details={"exception": repr(exc)},
        ).to_dict()

    fn = getattr(module, probe.target_symbol, None)
    if not callable(fn):
        return SmokeResult(
            status=SMOKE_STATUS_FAIL,
            target_symbol=target_symbol,
            probe_id=probe.probe_id,
            message="allowlisted smoke target is missing or not callable",
            details={"callable_name": probe.target_symbol},
        ).to_dict()

    try:
        probe.runner(fn)
    except Exception as exc:
        return SmokeResult(
            status=SMOKE_STATUS_FAIL,
            target_symbol=target_symbol,
            probe_id=probe.probe_id,
            message="allowlisted smoke probe failed",
            details={"exception": repr(exc)},
        ).to_dict()

    return SmokeResult(
        status=SMOKE_STATUS_PASS,
        target_symbol=target_symbol,
        probe_id=probe.probe_id,
        message="allowlisted smoke probe passed",
        details={},
    ).to_dict()
