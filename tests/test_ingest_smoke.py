"""Tests for ingest-time deterministic smoke validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciona.commands.ingest_cmds import _cmd_ingest
from sciona.ingester.smoke import run_smoke_validation


class _FakeEnv:
    async def close(self) -> None:
        return None


class _FakeCDG:
    def __init__(self) -> None:
        self.nodes = [{"id": "root"}]
        self.edges = []

    def model_dump(self) -> dict[str, object]:
        return {"nodes": self.nodes, "edges": self.edges}


class _FakeBundle:
    def __init__(self, *, generated_atoms: str) -> None:
        self.generated_atoms = generated_atoms
        self.generated_state_models = ""
        self.generated_witnesses = ""
        self.cdg = _FakeCDG()
        self.match_results: list[object] = []
        self.mypy_passed = True
        self.ghost_sim_passed = True


class _FakeAgent:
    bundle: _FakeBundle | None = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def ingest_state(self, source_path: str, class_name: str) -> dict[str, object]:
        assert self.bundle is not None
        return {"bundle": self.bundle, "error": ""}


def _patch_ingest_runtime(monkeypatch, tmp_path: Path) -> None:
    config = SimpleNamespace(
        ingester_llm_provider="tests",
        llm_provider="tests",
        ingester_llm_model="fixture",
        llm_model="fixture",
        ingester_max_depth=2,
        index_dir=tmp_path / "missing_index",
        ingester_decompose_line_threshold=30,
        ingester_shared_context_budget_chars=256,
        ingester_parallelism=1,
        ingester_cache_enabled=False,
        ingester_cache_dir=tmp_path / "cache",
    )
    mode_settings = SimpleNamespace(
        semantic_index_backend_override=None,
        ingester_shared_context_enabled=False,
    )

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_shared_context(*args, **kwargs):
        return None, None

    monkeypatch.setattr("sciona.config.AgeomConfig", lambda: config)
    monkeypatch.setattr(
        "sciona.config.resolve_execution_mode",
        lambda _config, _mode: mode_settings,
    )
    monkeypatch.setattr("sciona.ingester.IngesterAgent", _FakeAgent)
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._create_llm_router",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._warm_llm_if_supported",
        _noop_warm,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._create_proof_env",
        lambda *args, **kwargs: _FakeEnv(),
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._create_shared_context",
        _fake_shared_context,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._print_mode_summary",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._print_prompt_routing_summary",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._print_shared_context_metrics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sciona.commands.ingest_cmds._write_shared_context_metrics_file",
        lambda *args, **kwargs: None,
    )


async def _run_ingest(
    monkeypatch,
    tmp_path: Path,
    *,
    class_name: str,
    generated_atoms: str,
    output_parts: tuple[str, ...] = ("sklearn", "images"),
):
    _patch_ingest_runtime(monkeypatch, tmp_path)
    _FakeAgent.bundle = _FakeBundle(generated_atoms=generated_atoms)
    source_path = tmp_path / "source.py"
    source_path.write_text("def stub():\n    return None\n", encoding="utf-8")
    output_dir = tmp_path.joinpath(*output_parts)
    args = argparse.Namespace(
        source=str(source_path),
        class_name=class_name,
        output=str(output_dir),
        output_scope="family",
        llm_provider=None,
        llm_model=None,
        procedural=False,
        trace=False,
        monitor=False,
        stale_seconds=120,
        mode=None,
    )
    await _cmd_ingest(args)
    return output_dir


def _write_staged_atoms(
    tmp_path: Path,
    *,
    generated_atoms: str,
) -> Path:
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    (staged_dir / "atoms.py").write_text(generated_atoms, encoding="utf-8")
    return staged_dir


def test_run_smoke_validation_grouped_images_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "def grid_to_graph(n_x, n_y, n_z=1, **kwargs):\n"
            "    if n_x is None:\n"
            "        raise TypeError('n_x')\n"
            "    node_count = int(n_x) * int(n_y) * int(n_z)\n"
            "    return np.zeros((node_count, node_count))\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="images",
        target_symbol="grid_to_graph",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "sklearn.images.grid_to_graph.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_fft_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "def fft(x):\n"
            "    if x is None:\n"
            "        raise TypeError('x')\n"
            "    return np.fft.fft(x)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="signal_helpers",
        target_symbol="fft",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "numerical.fft.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_grouped_tempo_offsets_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "def utc_to_tai_leap_second_kernel(seconds, leap_seconds=37.0):\n"
            "    if seconds is None:\n"
            "        raise TypeError('seconds')\n"
            "    return float(seconds) + float(leap_seconds)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="offsets",
        target_symbol="utc_to_tai_leap_second_kernel",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "tempo_jl.offsets.utc_to_tai_leap_second_kernel.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_hamilton_segmentation_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "def hamilton_segmentation(signal, sampling_rate):\n"
            "    if signal is None:\n"
            "        raise TypeError('signal')\n"
            "    if not isinstance(sampling_rate, (int, float)):\n"
            "        raise TypeError('sampling_rate')\n"
            "    return np.array([300, 800, 1300, 1800], dtype=int)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="ecg_detectors",
        target_symbol="hamilton_segmentation",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "biosppy.ecg.hamilton_segmentation.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_detect_signal_onsets_elgendi2013_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "def detect_signal_onsets_elgendi2013(signal, sampling_rate, peakwindow, beatwindow, beatoffset, mindelay):\n"
            "    if not isinstance(sampling_rate, (int, float)):\n"
            "        raise TypeError('sampling_rate')\n"
            "    return np.array([50, 150, 250, 350, 450, 550], dtype=int)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="ppg_detectors",
        target_symbol="detect_signal_onsets_elgendi2013",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "biosppy.ppg.detect_signal_onsets_elgendi2013.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_threshold_based_onset_detection_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "def threshold_based_onset_detection(signal, rest, sampling_rate, threshold, active_state_duration):\n"
            "    if signal is None:\n"
            "        raise TypeError('signal')\n"
            "    return np.array([], dtype=int)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="emg_detectors",
        target_symbol="threshold_based_onset_detection",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "biosppy.emg.threshold_based_onset_detection.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_linalg_solve_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "import scipy.linalg\n"
            "def solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False):\n"
            "    if a is None or b is None:\n"
            "        raise TypeError('a or b')\n"
            "    return scipy.linalg.solve(a, b, lower=lower, overwrite_a=overwrite_a, overwrite_b=overwrite_b, check_finite=check_finite, assume_a=assume_a, transposed=transposed)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="linalg",
        target_symbol="solve",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "scipy.linalg.solve.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_linalg_lu_factor_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import scipy.linalg\n"
            "def lu_factor(a, overwrite_a=False, check_finite=True):\n"
            "    if a is None:\n"
            "        raise TypeError('a')\n"
            "    return scipy.linalg.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="linalg",
        target_symbol="lu_factor",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "scipy.linalg.lu_factor.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_optimize_minimize_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "import scipy.optimize\n"
            "def minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):\n"
            "    if fun is None or x0 is None:\n"
            "        raise TypeError('fun or x0')\n"
            "    return scipy.optimize.minimize(fun, x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="optimize",
        target_symbol="minimize",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "scipy.optimize.minimize.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_optimize_differential_evolution_passes(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "import scipy.optimize\n"
            "def differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1, popsize=5, tol=0.1, mutation=(0.5, 1.0), recombination=0.7, rng=None, callback=None, disp=False, polish=False, init='latinhypercube', atol=0.0, updating='immediate', workers=1, constraints=(), x0=None, integrality=None, vectorized=False):\n"
            "    if func is None or bounds is None:\n"
            "        raise TypeError('func or bounds')\n"
            "    return scipy.optimize.differential_evolution(func, bounds, args=args, strategy=strategy, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation, recombination=recombination, rng=rng, callback=callback, disp=disp, polish=polish, init=init, atol=atol, updating=updating, workers=workers, constraints=constraints, x0=x0, integrality=integrality, vectorized=vectorized)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="optimize",
        target_symbol="differential_evolution",
    )

    assert result["status"] == "pass"
    assert result["probe_id"] == "scipy.optimize.differential_evolution.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "pass"


def test_run_smoke_validation_hamilton_segmenter_fails_when_negative_path_does_not_raise(tmp_path: Path):
    staged_dir = _write_staged_atoms(
        tmp_path,
        generated_atoms=(
            "import numpy as np\n"
            "def hamilton_segmenter(signal, sampling_rate):\n"
            "    return np.array([300, 800, 1300], dtype=int)\n"
        ),
    )

    result = run_smoke_validation(
        staged_dir,
        package_basename="ecg_detectors",
        target_symbol="hamilton_segmenter",
    )

    assert result["status"] == "fail"
    assert result["probe_id"] == "biosppy.ecg.hamilton_segmenter.basic"
    assert result["details"]["positive_case"]["status"] == "pass"
    assert result["details"]["negative_case"]["status"] == "fail"
    assert result["details"]["negative_case"]["message"] == "negative-path probe did not raise"


@pytest.mark.asyncio
async def test_smoke_validation_not_applicable_allows_publication(monkeypatch, tmp_path: Path):
    output_dir = await _run_ingest(
        monkeypatch,
        tmp_path,
        class_name="ungrouped_atom",
        generated_atoms=(
            "def ungrouped_atom(x):\n"
            "    return x\n"
        ),
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["smoke_validation"]["status"] == "not_applicable"
    assert completed["summary"]["smoke_validation"]["status"] == "not_applicable"


@pytest.mark.asyncio
async def test_grouped_tempo_offsets_smoke_validation_passes_during_ingest(monkeypatch, tmp_path: Path):
    output_dir = await _run_ingest(
        monkeypatch,
        tmp_path,
        class_name="utc_to_tai_leap_second_kernel",
        generated_atoms=(
            "def utc_to_tai_leap_second_kernel(seconds, leap_seconds=37.0):\n"
            "    if seconds is None:\n"
            "        raise TypeError('seconds')\n"
            "    return float(seconds) + float(leap_seconds)\n"
        ),
        output_parts=("tempo_jl", "offsets"),
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["smoke_validation"]["status"] == "pass"
    assert status["smoke_validation"]["probe_id"] == "tempo_jl.offsets.utc_to_tai_leap_second_kernel.basic"
    assert completed["summary"]["smoke_validation"]["status"] == "pass"
    assert (output_dir / "atoms.py").exists()


@pytest.mark.asyncio
async def test_smoke_validation_pass_allows_publication(monkeypatch, tmp_path: Path):
    output_dir = await _run_ingest(
        monkeypatch,
        tmp_path,
        class_name="grid_to_graph",
        generated_atoms=(
            "import numpy as np\n"
            "def grid_to_graph(n_x, n_y, n_z=1, **kwargs):\n"
            "    if n_x is None:\n"
            "        raise TypeError('n_x')\n"
            "    node_count = int(n_x) * int(n_y) * int(n_z)\n"
            "    return np.zeros((node_count, node_count))\n"
        ),
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["smoke_validation"]["status"] == "pass"
    assert completed["summary"]["smoke_validation"]["status"] == "pass"
    assert (output_dir / "atoms.py").exists()


@pytest.mark.asyncio
async def test_smoke_validation_detector_pass_allows_publication(monkeypatch, tmp_path: Path):
    output_dir = await _run_ingest(
        monkeypatch,
        tmp_path,
        class_name="hamilton_segmentation",
        generated_atoms=(
            "import numpy as np\n"
            "def hamilton_segmentation(signal, sampling_rate):\n"
            "    if signal is None:\n"
            "        raise TypeError('signal')\n"
            "    if not isinstance(sampling_rate, (int, float)):\n"
            "        raise TypeError('sampling_rate')\n"
            "    return np.array([300, 800, 1300, 1800], dtype=int)\n"
        ),
        output_parts=("sciona", "biosppy", "ecg_detectors"),
    )

    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "COMPLETED.json").read_text(encoding="utf-8"))

    assert status["smoke_validation"]["status"] == "pass"
    assert status["smoke_validation"]["probe_id"] == "biosppy.ecg.hamilton_segmentation.basic"
    assert completed["summary"]["smoke_validation"]["status"] == "pass"
    assert (output_dir / "atoms.py").exists()


@pytest.mark.asyncio
async def test_smoke_validation_detector_fail_blocks_publication(monkeypatch, tmp_path: Path):
    _patch_ingest_runtime(monkeypatch, tmp_path)
    _FakeAgent.bundle = _FakeBundle(
        generated_atoms=(
            "import numpy as np\n"
            "def hamilton_segmentation(signal, sampling_rate):\n"
            "    return np.array([300, 800, 1300, 1800], dtype=int)\n"
        )
    )
    source_path = tmp_path / "source.py"
    source_path.write_text("def stub():\n    return None\n", encoding="utf-8")
    output_dir = tmp_path / "sciona" / "biosppy" / "ecg_detectors"

    with pytest.raises(SystemExit) as excinfo:
        await _cmd_ingest(
            argparse.Namespace(
                source=str(source_path),
                class_name="hamilton_segmentation",
                output=str(output_dir),
                output_scope="family",
                llm_provider=None,
                llm_model=None,
                procedural=False,
                trace=False,
                monitor=False,
                stale_seconds=120,
                mode=None,
            )
        )

    assert excinfo.value.code == 1
    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    failed = json.loads((output_dir / "FAILED.json").read_text(encoding="utf-8"))

    assert status["state"] == "failed"
    assert status["smoke_validation"]["status"] == "fail"
    assert status["smoke_validation"]["probe_id"] == "biosppy.ecg.hamilton_segmentation.basic"
    assert failed["summary"]["smoke_validation"]["status"] == "fail"
    assert failed["summary"]["published_files"] == []
    assert not (output_dir / "atoms.py").exists()


@pytest.mark.asyncio
async def test_smoke_validation_fail_blocks_publication(monkeypatch, tmp_path: Path):
    _patch_ingest_runtime(monkeypatch, tmp_path)
    _FakeAgent.bundle = _FakeBundle(
        generated_atoms=(
            "import numpy as np\n"
            "def grid_to_graph(n_x, n_y, n_z=1, **kwargs):\n"
            "    node_count = 3\n"
            "    return np.zeros((node_count, node_count))\n"
        )
    )
    source_path = tmp_path / "source.py"
    source_path.write_text("def stub():\n    return None\n", encoding="utf-8")
    output_dir = tmp_path / "sklearn" / "images"

    with pytest.raises(SystemExit) as excinfo:
        await _cmd_ingest(
            argparse.Namespace(
                source=str(source_path),
                class_name="grid_to_graph",
                output=str(output_dir),
                output_scope="family",
                llm_provider=None,
                llm_model=None,
                procedural=False,
                trace=False,
                monitor=False,
                stale_seconds=120,
                mode=None,
            )
        )

    assert excinfo.value.code == 1
    status = json.loads((output_dir / ".ingest_status.json").read_text(encoding="utf-8"))
    failed = json.loads((output_dir / "FAILED.json").read_text(encoding="utf-8"))

    assert status["state"] == "failed"
    assert status["smoke_validation"]["status"] == "fail"
    assert failed["summary"]["smoke_validation"]["status"] == "fail"
    assert failed["summary"]["published_files"] == []
    assert not (output_dir / "atoms.py").exists()
