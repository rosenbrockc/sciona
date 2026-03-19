"""Tests for the CDG pre-screen gate."""

from __future__ import annotations

import pytest

from ageom.clearinghouse.prescreen import prescreen, _scan_source, _check_structure


class TestSuspiciousPatternScan:
    def test_clean_source_passes(self):
        source = "import numpy as np\ndef f(x): return np.sum(x)"
        assert _scan_source(source) == []

    def test_socket_import_rejected(self):
        source = "import socket\ndef f(): socket.connect(('evil.com', 80))"
        reasons = _scan_source(source)
        assert any("socket" in r for r in reasons)

    def test_urllib_import_rejected(self):
        source = "from urllib.request import urlopen"
        reasons = _scan_source(source)
        assert any("urllib" in r for r in reasons)

    def test_requests_import_rejected(self):
        source = "import requests"
        reasons = _scan_source(source)
        assert any("requests" in r for r in reasons)

    def test_exec_call_rejected(self):
        source = "exec('import os')"
        reasons = _scan_source(source)
        assert any("exec" in r for r in reasons)

    def test_eval_call_rejected(self):
        source = "x = eval('1+1')"
        reasons = _scan_source(source)
        assert any("eval" in r for r in reasons)

    def test_subprocess_import_rejected(self):
        source = "import subprocess"
        reasons = _scan_source(source)
        assert any("subprocess" in r for r in reasons)

    def test_os_system_rejected(self):
        source = "import os\nos.system('rm -rf /')"
        reasons = _scan_source(source)
        assert any("os.system" in r for r in reasons)

    def test_ctypes_rejected(self):
        source = "import ctypes"
        reasons = _scan_source(source)
        assert any("ctypes" in r for r in reasons)

    def test_open_write_mode_rejected(self):
        source = "f = open('/etc/passwd', 'w')"
        reasons = _scan_source(source)
        assert any("write" in r.lower() or "open" in r.lower() for r in reasons)

    def test_pathlib_write_rejected(self):
        source = "from pathlib import Path\nPath('x').write_text('hack')"
        reasons = _scan_source(source)
        assert any("write_text" in r for r in reasons)

    def test_open_read_mode_ok(self):
        source = "f = open('data.csv', 'r')"
        reasons = _scan_source(source)
        assert reasons == []

    def test_syntax_error_rejected(self):
        source = "def f(: pass"
        reasons = _scan_source(source)
        assert any("SyntaxError" in r for r in reasons)

    def test_compile_call_rejected(self):
        source = "compile('print(1)', '<string>', 'exec')"
        reasons = _scan_source(source)
        assert any("compile" in r for r in reasons)

    def test_importlib_rejected(self):
        source = "import importlib"
        reasons = _scan_source(source)
        assert any("importlib" in r for r in reasons)


class TestStructuralValidity:
    def test_valid_dag(self):
        reasons = _check_structure(["a", "b", "c"], [("a", "b"), ("b", "c")])
        assert reasons == []

    def test_cycle_detected(self):
        reasons = _check_structure(["a", "b"], [("a", "b"), ("b", "a")])
        assert any("cycle" in r.lower() for r in reasons)

    def test_unknown_atom_rejected(self):
        known = frozenset({"a", "b"})
        reasons = _check_structure(["a", "c"], [], known_atoms=known)
        assert any("Unknown atom" in r for r in reasons)

    def test_skip_registry_check(self):
        reasons = _check_structure(["a", "b"], [])
        assert reasons == []


class TestFullPrescreen:
    def test_valid_cdg_passes(self):
        sources = {
            "pkg.filter": "import numpy\ndef f(x): return x[x > 0]",
            "pkg.sort": "def g(x): return sorted(x)",
        }
        result = prescreen(sources, [("pkg.filter", "pkg.sort")])
        assert result.passed
        assert result.rejection_reasons == []
        assert result.estimated_tier in ("standard", "heavy", "gpu")

    def test_malicious_source_fails(self):
        sources = {"pkg.evil": "import socket\ndef f(): pass"}
        result = prescreen(sources)
        assert not result.passed
        assert len(result.rejection_reasons) > 0

    def test_cyclic_dag_fails(self):
        sources = {
            "a": "def f(): pass",
            "b": "def g(): pass",
        }
        result = prescreen(sources, [("a", "b"), ("b", "a")])
        assert not result.passed

    def test_too_many_nodes_fails(self):
        sources = {f"atom_{i}": "def f(): pass" for i in range(150)}
        result = prescreen(sources, max_nodes=100)
        assert not result.passed
        assert any("Node count" in r for r in result.rejection_reasons)

    def test_resource_estimation_standard(self):
        sources = {f"a{i}": "def f(): pass" for i in range(5)}
        result = prescreen(sources)
        assert result.passed
        assert result.estimated_tier == "standard"

    def test_resource_estimation_heavy(self):
        sources = {f"a{i}": "def f(): pass" for i in range(40)}
        result = prescreen(sources)
        assert result.passed
        assert result.estimated_tier == "heavy"

    def test_fail_fast_on_pattern_scan(self):
        sources = {
            "pkg.bad": "import socket",
            "pkg.good": "def f(): pass",
        }
        result = prescreen(
            sources,
            [("pkg.bad", "pkg.good")],
            known_atoms=frozenset({"pkg.bad"}),  # pkg.good not known
        )
        # Should fail on pattern scan, not reach structural check
        assert not result.passed
        assert all("Unknown atom" not in r for r in result.rejection_reasons)
