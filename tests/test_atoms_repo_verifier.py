from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from sciona.atoms_repo_verifier import verify_atoms_repo


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def test_verifier_reports_actionable_missing_symbols(tmp_path: Path) -> None:
    repo = tmp_path / "atoms_repo"
    pkg = repo / "fakeatoms"

    _write(pkg / "__init__.py", "")
    _write(
        pkg / "ecg.py",
        """
        from __future__ import annotations

        from sciona.ghost.registry import register_atom

        @register_atom(witness_bandpass_filter)
        def bandpass_filter(signal: np.ndarray, state: ECGPipelineState) -> tuple[np.ndarray, ECGPipelineState]:
            obj = ECGProcessor.__new__(ECGProcessor)
            return signal, state
        """,
    )
    _write(
        pkg / "ecg_witnesses.py",
        """
        def witness_bandpass_filter(signal, state):
            return signal, state
        """,
    )
    _write(
        pkg / "ecg_state.py",
        """
        class ECGPipelineState:
            pass
        """,
    )

    report = verify_atoms_repo(repo, "fakeatoms")

    symbols = {issue.symbol: issue for issue in report.issues}
    assert "witness_bandpass_filter" in symbols
    assert symbols["witness_bandpass_filter"].hint == (
        "Import `witness_bandpass_filter` from sibling module `ecg_witnesses`."
    )
    assert "ECGPipelineState" in symbols
    assert symbols["ECGPipelineState"].hint == (
        "Import `ECGPipelineState` from sibling module `ecg_state`."
    )
    assert "ECGProcessor" in symbols
    assert symbols["ECGProcessor"].hint == (
        "Define the symbol or import it explicitly in this module."
    )
    assert "np" in symbols
    assert symbols["np"].hint == "Add `import numpy as np`."


def test_verifier_reports_syntax_errors(tmp_path: Path) -> None:
    repo = tmp_path / "atoms_repo"
    pkg = repo / "fakeatoms"
    _write(pkg / "__init__.py", "")
    _write(
        pkg / "broken.py",
        """
        def nope(:
            pass
        """,
    )

    report = verify_atoms_repo(repo, "fakeatoms")

    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.code == "syntax-error"
    assert "invalid syntax" in issue.context


def test_script_json_output(tmp_path: Path) -> None:
    repo = tmp_path / "atoms_repo"
    pkg = repo / "fakeatoms"
    _write(pkg / "__init__.py", "")
    _write(
        pkg / "ecg.py",
        """
        from sciona.ghost.registry import register_atom

        @register_atom(witness_bandpass_filter)
        def bandpass_filter(signal):
            return signal
        """,
    )

    script = Path("scripts/verify_atoms_repo.py").resolve()
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(repo),
            "--package",
            "fakeatoms",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["issue_count"] >= 1
    assert payload["issues"][0]["code"] == "undefined-name"


def test_verifier_does_not_flag_comprehension_or_loop_locals(tmp_path: Path) -> None:
    repo = tmp_path / "atoms_repo"
    pkg = repo / "fakeatoms"
    _write(pkg / "__init__.py", "")
    _write(
        pkg / "sorting.py",
        """
        import icontract

        @icontract.ensure(lambda result: all(result[i] <= result[i + 1] for i in range(len(result) - 1)))
        def sort_vals(result):
            total = 0
            for value in result:
                total += value
            return total
        """,
    )

    report = verify_atoms_repo(repo, "fakeatoms")

    assert report.ok


def test_verifier_supports_dotted_namespace_package_paths(tmp_path: Path) -> None:
    repo = tmp_path / "atoms_repo"
    pkg = repo / "sciona" / "atoms" / "demo"
    _write(repo / "sciona" / "__init__.py", "")
    _write(repo / "sciona" / "atoms" / "__init__.py", "")
    _write(pkg / "__init__.py", "")
    _write(
        pkg / "filters.py",
        """
        def bandpass_filter(signal):
            return signal
        """,
    )

    report = verify_atoms_repo(repo, "sciona.atoms.demo", import_smoke=True)

    assert report.ok
