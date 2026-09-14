"""Build pinned public ARC software and exercise only generated synthetic tasks.

This is preliminary execution evidence, not a publication or accuracy review.
The source cache is read only; portability edits exist only in a temporary copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
UNITS = "read core_functions image_functions image_functions2 visu normalize tasks runner score load evals brute2 deduce_op pieces compose2 brute_size efficient main".split()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--compiler", default="clang++")
    args = parser.parse_args()
    pins_path = ROOT / "docs/reviews/competition_arc_source_pins.json"
    pins = json.loads(pins_path.read_text())
    verified = {}
    for entry in pins["files"]:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe source pin")
        data = (args.source_cache / rel).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if sha(data) != entry["sha256"] or blob != entry["git_blob_sha1"]:
            raise ValueError(f"Source mismatch: {rel}")
        verified[rel.as_posix()] = data
    compiler = subprocess.run([args.compiler, "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    patches = []
    checks = []
    with tempfile.TemporaryDirectory(prefix="sciona_arc_validation_") as temp:
        build = Path(temp) / "build"
        build.mkdir()
        for rel, data in verified.items():
            if Path(rel).parts[0] not in {"headers", "src"}:
                continue
            dest = build / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        for rel, replacements in {
            "src/read.cpp": [("<experimental/filesystem>", "<filesystem>"), ("experimental::filesystem::", "filesystem::")],
            "headers/precompiled_stl.hpp": [("#include <vector>", "#include <map>\n#include <chrono>\n#include <vector>")],
        }.items():
            path = build / rel
            before = path.read_bytes()
            text = before.decode()
            for old, new in replacements:
                if old not in text:
                    raise ValueError("Expected portability patch anchor missing")
                text = text.replace(old, new)
            after = text.encode()
            path.write_bytes(after)
            patches.append({"path": rel, "before_sha256": sha(before), "after_sha256": sha(after), "replacements": replacements})
        flags = ["-std=c++17", "-O2", "-fsigned-char", "-Iheaders"]
        result = subprocess.run([args.compiler, *flags, *[f"src/{name}.cpp" for name in UNITS], "-o", "run"], cwd=build, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr[-8000:])
        warning_count = result.stderr.count("warning:")
        binary_sha256 = sha((build / "run").read_bytes())
        # All grids below are invented here; no competition task files are read.
        cases = {
            "identity": (
                [([[0, 1], [1, 0]], [[0, 1], [1, 0]]), ([[2, 0], [0, 2]], [[2, 0], [0, 2]])],
                [[0, 3], [3, 0]], [[0, 3], [3, 0]],
            ),
            "transpose": (
                [([[1, 0, 0], [0, 1, 0]], [[1, 0], [0, 1], [0, 0]]),
                 ([[0, 2, 0], [0, 0, 2]], [[0, 0], [2, 0], [0, 2]])],
                [[3, 3, 0], [0, 0, 3]], [[3, 0], [3, 0], [0, 3]],
            ),
        }
        for alias, (training, test, expected) in cases.items():
            for mode in [2, 3, 23, 33, 4, 12]:
                work = Path(temp) / f"{alias}_{mode}"
                task_dir = work / "dataset/evaluation"
                task_dir.mkdir(parents=True)
                (work / "output").mkdir()
                task = {"train": [{"input": x, "output": y} for x, y in training], "test": [{"input": test}]}
                (task_dir / "00000000.json").write_text(json.dumps(task))
                start = time.monotonic()
                run = subprocess.run([str(build / "run"), "0", str(mode)], cwd=work, capture_output=True, text=True, timeout=60)
                if run.returncode:
                    raise RuntimeError(f"Synthetic {alias}/{mode} failed: {run.stderr[-1500:]}")
                lines = (work / f"output/answer_0_{mode}.csv").read_text().splitlines()
                if lines[0] != "00000000_0" or not 1 <= len(lines[1:]) <= 3:
                    raise AssertionError("Bad source answer envelope")
                predictions, scores = [], []
                for line in lines[1:]:
                    grid, score = line.split()
                    rows = [[int(x) for x in row] for row in grid.strip("|").split("|")]
                    if not 1 <= len(rows) <= 30 or not 1 <= len(rows[0]) <= 30 or any(len(row) != len(rows[0]) for row in rows):
                        raise AssertionError("Bad source answer grid")
                    predictions.append(rows)
                    scores.append(float(score))
                if not all(math.isfinite(score) for score in scores):
                    raise AssertionError("Nonfinite candidate score")
                checks.append({"synthetic_case": alias, "argument": mode, "search_cost_limit": mode % 10 * 10,
                               "candidate_count": len(predictions), "expected_in_top_three": expected in predictions,
                               "expected_rank": predictions.index(expected) + 1 if expected in predictions else None,
                               "seconds": round(time.monotonic() - start, 4), "scores": scores,
                               "test_output_omitted": True})
                print(f"{alias}/{mode}: expected_in_top_three={expected in predictions}", flush=True)
        if not all(item["expected_in_top_three"] for item in checks):
            raise AssertionError("A synthetic transformation was not recovered")
    report = {
        "format": "arc-source-execution.v1", "status": "preliminary_synthetic_execution_passed",
        "source_commit": pins["commit"], "source_pins_sha256": sha(pins_path.read_bytes()),
        "validator_sha256": sha(Path(__file__).read_bytes()), "verified_source_files": len(verified),
        "compiler": compiler, "flags": flags, "translation_units": UNITS,
        "portability_patches": patches, "compiler_warning_count": warning_count,
        "observed_binary_sha256": binary_sha256, "checks": checks,
        "limits": ["Synthetic checks establish execution only, not competition accuracy or Tier 3 approval.",
                   "Original evaluation runner used with absent test labels; its diagnostic verdict is not an inference correctness measure.",
                   "No original nine-hour adaptive safe_run scheduler executed or reproduced.",
                   "No historical GCC binary equivalence, sanitizer clearance, provider contract, or served graph validation established.",
                   "Signed char explicitly preserves negative image sentinels and parser EOF on ARM."]}
    path = ROOT / "docs/reviews/competition_arc_source_execution.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Recorded", path.name)


if __name__ == "__main__":
    main()
