"""Tests for discipline repo webhook synchronization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ageom.ecosystem.models import DisciplineRepo
from ageom.ecosystem.webhook_sync import (
    diff_atoms,
    parse_manifest_sqlite,
    should_sync,
    validate_webhook_signature,
)


class TestWebhookSignature:
    def test_valid_signature(self):
        payload = b'{"ref": "refs/heads/main"}'
        secret = "my_secret"
        import hashlib, hmac
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert validate_webhook_signature(payload, sig, secret) is True

    def test_invalid_signature(self):
        payload = b'{"ref": "refs/heads/main"}'
        assert validate_webhook_signature(payload, "sha256=wrong", "secret") is False

    def test_missing_prefix(self):
        assert validate_webhook_signature(b"data", "noprefixhere", "secret") is False


class TestParseManifest:
    def test_parse_atoms(self, tmp_path: Path):
        db_path = tmp_path / "manifest.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("""CREATE TABLE atoms (
            atom_id TEXT, fqdn TEXT, status TEXT, domain_tags TEXT, description TEXT
        )""")
        con.execute(
            "INSERT INTO atoms VALUES (?, ?, ?, ?, ?)",
            ("a1", "pkg.filter", "approved", "signal,audio", "A filter"),
        )
        con.execute(
            "INSERT INTO atoms VALUES (?, ?, ?, ?, ?)",
            ("a2", "pkg.sort", "approved", "", "Sort function"),
        )
        con.commit()
        con.close()

        atoms = parse_manifest_sqlite(db_path)
        assert len(atoms) == 2
        assert atoms[0]["fqdn"] == "pkg.filter"
        assert atoms[0]["domain_tags"] == ["signal", "audio"]
        assert atoms[1]["fqdn"] == "pkg.sort"
        assert atoms[1]["domain_tags"] == []

    def test_missing_file(self, tmp_path: Path):
        atoms = parse_manifest_sqlite(tmp_path / "nonexistent.sqlite")
        assert atoms == []

    def test_no_atoms_table(self, tmp_path: Path):
        db_path = tmp_path / "manifest.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE other (id TEXT)")
        con.commit()
        con.close()
        atoms = parse_manifest_sqlite(db_path)
        assert atoms == []


class TestDiffAtoms:
    def test_new_atoms(self):
        local = [
            {"fqdn": "pkg.new_atom", "status": "approved"},
            {"fqdn": "pkg.existing", "status": "approved"},
        ]
        global_fqdns = frozenset({"pkg.existing"})
        new, updated = diff_atoms(local, global_fqdns)
        assert len(new) == 1
        assert new[0]["fqdn"] == "pkg.new_atom"
        assert len(updated) == 1

    def test_all_new(self):
        local = [{"fqdn": "pkg.a"}, {"fqdn": "pkg.b"}]
        new, updated = diff_atoms(local, frozenset())
        assert len(new) == 2
        assert len(updated) == 0

    def test_all_existing(self):
        local = [{"fqdn": "pkg.a"}]
        new, updated = diff_atoms(local, frozenset({"pkg.a"}))
        assert len(new) == 0
        assert len(updated) == 1

    def test_empty_local(self):
        new, updated = diff_atoms([], frozenset({"pkg.a"}))
        assert new == []
        assert updated == []


class TestShouldSync:
    def test_new_commit(self):
        repo = DisciplineRepo(repo_url="https://github.com/org/repo", last_synced_commit="abc")
        assert should_sync(repo, "def") is True

    def test_same_commit(self):
        repo = DisciplineRepo(repo_url="https://github.com/org/repo", last_synced_commit="abc")
        assert should_sync(repo, "abc") is False

    def test_first_sync(self):
        repo = DisciplineRepo(repo_url="https://github.com/org/repo", last_synced_commit="")
        assert should_sync(repo, "abc") is True
