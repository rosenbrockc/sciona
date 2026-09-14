"""Synthetic scheduling/resource tests; no competition tasks or benchmarks."""
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from sciona.arc_contract import parse_answers, validate_task
from sciona.arc_process import Attempt, ProcessBudget
from sciona.arc_schedule import commands_for_phase, run_schedule


def task(count=5):
    return validate_task({"train": [{"input": [[1]], "output": [[1]]}],
                          "test": [{"input": [[2]]} for _ in range(count)]})


def test_source_estimates_slack_stable_order():
    baseline = {0: Attempt("success", 3, 2, 8, ()),
                1: Attempt("memory_limit", 3, 1, 4, ()),
                2: Attempt("success", 3, 1, 5, ())}
    first, concurrency = commands_for_phase(3, 3, {}, 100, 200)
    assert concurrency == 4
    assert [(x.expected_seconds, x.expected_memory_mib, x.slack) for x in first] == [(100, 200, 1.5)] * 3
    for mode in [23, 33, 4]:
        commands, concurrency = commands_for_phase(mode, 3, baseline, 100, 200)
        assert [x.test_index for x in commands] == [1, 2, 0]
        scale, slack = (20, 2) if mode == 4 else (2, 100)
        assert concurrency == (2 if mode == 4 else 4)
        assert [(x.expected_seconds, x.expected_memory_mib, x.slack) for x in commands] == [
            (scale, 4 * scale, slack), (scale, 5 * scale, slack), (2 * scale, 8 * scale, slack)]


def test_shared_budget_marks_largest_and_releases():
    budget = ProcessBudget(30, 10)
    assert budget.observe("large", 8) is None
    assert budget.observe("small", 3) is None
    assert budget.observe("large", 8) == "memory_limit"
    budget.release("large")
    assert budget.observe("small", 3) is None
    budget.cancel()
    assert budget.observe("small", 0) == "cancelled"


def test_global_deadline_and_invalid_budget():
    with patch("sciona.arc_process.time.monotonic", return_value=0):
        budget = ProcessBudget(1, 10)
    with patch("sciona.arc_process.time.monotonic", return_value=1):
        assert budget.observe(object(), 0) == "time_limit"
    for value in [0, -1, True, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            ProcessBudget(value, 10)


def test_phases_concurrency_and_merging():
    lock = threading.Lock()
    active, peak, calls = {}, {}, []
    answers = parse_answers("00000000_0\n|2| 1\n")

    def worker(binary, digest, validated, index, argument, **kwargs):
        with lock:
            assert not any(count for mode, count in active.items() if mode != argument)
            active[argument] = active.get(argument, 0) + 1
            peak[argument] = max(peak.get(argument, 0), active[argument])
            calls.append((index, argument, kwargs["timeout_seconds"], kwargs["memory_mib"]))
        time.sleep(.015)
        with lock:
            active[argument] -= 1
        return Attempt("success", argument, index + 1, 10, answers)

    with patch("sciona.arc_schedule.run_once", side_effect=worker):
        result = run_schedule(Path("unused"), "unused", task(), seconds=100, memory_mib=200)
    assert result.completed_all_runs and not result.missing_test_indices
    assert len(result.attempts) == 20
    assert peak == {3: 4, 23: 4, 33: 4, 4: 2}
    for index, argument, seconds, memory in calls:
        if argument == 3:
            assert (seconds, memory) == (150, 300)
        elif argument in [23, 33]:
            assert (seconds, memory) == ((index + 1) * 200, 2000)
        else:
            assert (seconds, memory) == ((index + 1) * 40, 400)


def test_missing_runs_never_become_successful_fallback():
    def worker(binary, digest, validated, index, argument, **kwargs):
        return Attempt("memory_limit", argument, .01, 1, ())
    with patch("sciona.arc_schedule.run_once", side_effect=worker):
        result = run_schedule(Path("unused"), "unused", task(2), seconds=30, memory_mib=10)
    assert not result.completed_all_runs
    assert result.missing_test_indices == (0, 1)
    assert result.predictions == ((), ())


def test_exception_cancels_other_workers_before_join():
    other_started = threading.Event()
    cancellation_seen = threading.Event()

    def worker(binary, digest, validated, index, argument, *, budget, **kwargs):
        if index == 0:
            assert other_started.wait(2)
            raise RuntimeError("synthetic failure")
        other_started.set()
        until = time.monotonic() + 2
        while time.monotonic() < until:
            if budget.observe("worker", 0) == "cancelled":
                cancellation_seen.set()
                return Attempt("cancelled", argument, 0, 0, ())
            time.sleep(.001)
        raise AssertionError("Worker was not cancelled before executor join")

    with patch("sciona.arc_schedule.run_once", side_effect=worker), pytest.raises(RuntimeError, match="synthetic failure"):
        run_schedule(Path("unused"), "unused", task(2), seconds=30, memory_mib=10)
    assert cancellation_seen.is_set()
