"""Strict Python boundary for the pinned top-quarks ARC C++ solver.

Source algorithm and ranking: MIT licensed top-quarks/ARC-solution at
9407072659de1270358c2ba34c527785214dd68b. See docs/licenses/ARC-solution-MIT.txt.
The cardinality caps here are explicit adapter resource limits, not source rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re


MAX_TRAIN_PAIRS = 20
MAX_TEST_INPUTS = 20
Grid = tuple[tuple[int, ...], ...]


def _grid(value: object) -> Grid:
    if type(value) is not list or not 1 <= len(value) <= 30:
        raise ValueError("Grid must contain 1..30 rows")
    width = None
    result = []
    for row in value:
        if type(row) is not list or not 1 <= len(row) <= 30:
            raise ValueError("Grid row must contain 1..30 colors")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("Grid must be rectangular")
        if any(type(color) is not int or not 0 <= color <= 9 for color in row):
            raise ValueError("Grid colors must be integers in 0..9")
        result.append(tuple(row))
    return tuple(result)


def _keys(value: object, expected: set[str]) -> dict:
    if type(value) is not dict or value.keys() != expected:
        raise ValueError("Unexpected object fields")
    return value


@dataclass(frozen=True)
class Task:
    train: tuple[tuple[Grid, Grid], ...]
    test: tuple[Grid, ...]

    def source_task(self, test_index: int) -> dict:
        """One test per isolated invocation avoids source output overwrites."""
        if type(test_index) is not int or not 0 <= test_index < len(self.test):
            raise ValueError("Invalid test index")
        def lists(grid: Grid) -> list[list[int]]:
            return [list(row) for row in grid]
        return {"train": [{"input": lists(x), "output": lists(y)} for x, y in self.train],
                "test": [{"input": lists(self.test[test_index])}]}


def validate_task(value: object) -> Task:
    value = _keys(value, {"train", "test"})
    train, test = value["train"], value["test"]
    if type(train) is not list or not 1 <= len(train) <= MAX_TRAIN_PAIRS:
        raise ValueError("Training pair count outside adapter limits")
    if type(test) is not list or not 1 <= len(test) <= MAX_TEST_INPUTS:
        raise ValueError("Test input count outside adapter limits")
    pairs = []
    inputs = []
    for pair in train:
        pair = _keys(pair, {"input", "output"})
        pairs.append((_grid(pair["input"]), _grid(pair["output"])))
    for item in test:
        item = _keys(item, {"input"})
        inputs.append(_grid(item["input"]))
    return Task(tuple(pairs), tuple(inputs))


@dataclass(frozen=True)
class Candidate:
    serialized: str
    grid: Grid
    score: float


def parse_answers(text: str) -> tuple[Candidate, ...]:
    """Read one isolated source answer, rejecting corrupt/truncated output."""
    if type(text) is not str or len(text) > 4096:
        raise ValueError("Invalid answer text")
    lines = text.splitlines()
    if not 2 <= len(lines) <= 4 or lines[0] != "00000000_0":
        raise ValueError("Invalid answer envelope")
    result = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Invalid answer record")
        serialized, score_text = parts
        if re.fullmatch(r"\|[0-9]{1,30}(?:\|[0-9]{1,30}){0,29}\|", serialized) is None:
            raise ValueError("Invalid serialized grid")
        grid = _grid([[int(color) for color in row] for row in serialized[1:-1].split("|")])
        score = float(score_text)
        if not math.isfinite(score):
            raise ValueError("Nonfinite candidate score")
        result.append(Candidate(serialized, grid, score))
    return tuple(result)


def merge_answers(runs: list[tuple[Candidate, ...]]) -> tuple[Candidate, ...]:
    """safe_run.py descending (score, image) ordering and exact deduplication.

No successful runs is a runtime failure. The source's empty-candidate fallback
is valid only when there was a completed output envelope.
"""
    if not runs:
        raise ValueError("No completed solver runs")
    ordered = sorted((candidate for run in runs for candidate in run),
                     key=lambda candidate: (candidate.score, candidate.serialized), reverse=True)
    best = []
    seen = set()
    for candidate in ordered:
        if candidate.serialized not in seen:
            best.append(candidate)
            seen.add(candidate.serialized)
            if len(best) == 3:
                break
    if not best:
        best.append(Candidate("|0|", ((0,),), -1.0))
    return tuple(best)
