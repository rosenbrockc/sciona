#!/usr/bin/env python3
"""Prepare /tmp scratch clones of sibling atoms repos for worker-safe editing."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SCRATCH_ROOT = Path("/tmp/sciona-atoms-worker-scratch")


@dataclass
class ScratchRepoReport:
    repo_name: str
    source_repo: str
    scratch_repo: str
    branch: str
    head: str
    dirty_paths: int
    mirrored_paths: int


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def parse_porcelain_z(repo: Path) -> list[tuple[str, str, str | None]]:
    raw = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        check=True,
        capture_output=True,
    ).stdout
    if not raw:
        return []

    entries = raw.decode("utf-8", errors="replace").split("\0")
    if entries and entries[-1] == "":
        entries.pop()

    parsed: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        status = entry[:2]
        path = entry[3:]
        second_path: str | None = None
        if "R" in status or "C" in status:
            index += 1
            second_path = entries[index]
        parsed.append((status, path, second_path))
        index += 1
    return parsed


def ensure_repo(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Source repo does not exist: {path}")
    git_dir = path / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(f"Source repo is not a git repository: {path}")


def delete_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def copy_path(source: Path, dest: Path) -> None:
    if source.is_symlink():
        delete_path(dest)
        target = source.readlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(target)
        return
    if source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def mirror_working_tree(source_repo: Path, scratch_repo: Path) -> tuple[int, int]:
    changes = parse_porcelain_z(source_repo)
    mirrored = 0

    for status, path, second_path in changes:
        rel_path = Path(second_path or path)
        source_path = source_repo / rel_path
        scratch_path = scratch_repo / rel_path

        if "R" in status and second_path is not None:
            delete_path(scratch_repo / path)

        if "D" in status and "R" not in status:
            delete_path(scratch_repo / path)
            mirrored += 1
            continue

        if source_path.exists() or source_path.is_symlink():
            copy_path(source_path, scratch_path)
            mirrored += 1

    return len(changes), mirrored


def prepare_repo(source_repo: Path, scratch_root: Path, refresh: bool) -> ScratchRepoReport:
    ensure_repo(source_repo)
    scratch_repo = scratch_root / source_repo.name

    if scratch_repo.exists():
        if not refresh:
            raise FileExistsError(
                f"Scratch repo already exists: {scratch_repo}. Use --refresh to recreate it."
            )
        shutil.rmtree(scratch_repo)

    scratch_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source_repo), str(scratch_repo)],
        check=True,
    )
    dirty_paths, mirrored_paths = mirror_working_tree(source_repo, scratch_repo)
    branch = run(["git", "-C", str(source_repo), "rev-parse", "--abbrev-ref", "HEAD"])
    head = run(["git", "-C", str(source_repo), "rev-parse", "HEAD"])
    return ScratchRepoReport(
        repo_name=source_repo.name,
        source_repo=str(source_repo.resolve()),
        scratch_repo=str(scratch_repo.resolve()),
        branch=branch,
        head=head,
        dirty_paths=dirty_paths,
        mirrored_paths=mirrored_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create isolated /tmp clones of sibling atoms repos so workers can edit them "
            "without touching the protected source repos."
        )
    )
    parser.add_argument(
        "repos",
        nargs="+",
        help="Source repo paths, for example /Users/conrad/personal/sciona-atoms-fintech.",
    )
    parser.add_argument(
        "--scratch-root",
        default=str(DEFAULT_SCRATCH_ROOT),
        help=f"Root directory for scratch clones. Default: {DEFAULT_SCRATCH_ROOT}",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recreate any existing scratch repo with the same name.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    scratch_root = Path(args.scratch_root).expanduser().resolve()
    reports = [
        prepare_repo(Path(repo).expanduser().resolve(), scratch_root, refresh=args.refresh)
        for repo in args.repos
    ]

    if args.json_output:
        print(json.dumps([asdict(report) for report in reports], indent=2))
    else:
        for report in reports:
            print(f"{report.repo_name}: {report.scratch_repo}")
            print(f"  source: {report.source_repo}")
            print(f"  branch: {report.branch}")
            print(f"  head: {report.head}")
            print(f"  dirty paths mirrored: {report.mirrored_paths}/{report.dirty_paths}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
