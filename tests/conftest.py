from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest

from tests.helpers.match_regression import (
    MatchCase,
    build_ageo_atoms_declarations,
    load_match_cases,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _check_llama_ready(base_url: str) -> bool:
    for endpoint in ("/models", "/health"):
        url = f"{base_url}{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            continue
    return False


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ageo_atoms_root(repo_root: Path) -> Path:
    configured = os.environ.get("AGEOM_TEST_AGEO_ATOMS_ROOT", "")
    if configured:
        return Path(configured).expanduser().resolve()
    return (repo_root / ".." / "ageo-atoms").resolve()


@pytest.fixture(scope="session")
def match_cases(repo_root: Path) -> list[MatchCase]:
    fixture_path = repo_root / "tests" / "fixtures" / "match_cases.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Missing match-case fixture: {fixture_path}")
    return load_match_cases(repo_root, fixture_path)


@pytest.fixture(scope="session")
def ageo_atoms_declarations(ageo_atoms_root: Path) -> list:
    if not ageo_atoms_root.exists():
        pytest.skip(f"ageo-atoms repo not found at {ageo_atoms_root}")
    return build_ageo_atoms_declarations(ageo_atoms_root)


@pytest.fixture(scope="session")
def llama_server() -> Iterator[dict[str, str]]:
    """Start a dedicated local llama.cpp server on an ephemeral port for live tests."""
    cmd_template = os.environ.get("AGEOM_TEST_LLAMA_SERVER_CMD", "").strip()
    if not cmd_template:
        pytest.skip(
            "Set AGEOM_TEST_LLAMA_SERVER_CMD to enable live llama regression tests"
        )

    model_name = os.environ.get("AGEOM_TEST_LLAMA_MODEL", "").strip() or "local-llama"

    port = _find_free_port()
    cmd_str = cmd_template.format(port=port)
    cmd = shlex.split(cmd_str)

    if "{port}" not in cmd_template and "--port" not in cmd and "-p" not in cmd:
        cmd.extend(["--host", "127.0.0.1", "--port", str(port)])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )

    base_url = f"http://127.0.0.1:{port}/v1"
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.skip("llama test server exited before becoming ready")
        if _check_llama_ready(base_url):
            break
        time.sleep(0.5)
    else:
        _terminate_process(proc)
        pytest.skip("llama test server did not become ready within 60s")

    try:
        yield {
            "base_url": base_url,
            "model": model_name,
            "port": str(port),
            "command": json.dumps(cmd),
        }
    finally:
        _terminate_process(proc)
