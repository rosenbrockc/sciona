"""Synthetic runtime envelope checks; no real task data."""
import os
from unittest.mock import patch

import pytest

from sciona.arc_runtime import prepare, reviewed_binary, execute


def payload():
    return {"version": 1, "task": {"train": [{"input": [[1]], "output": [[1]]}],
                                   "test": [{"input": [[2]]}]},
            "budget": {"seconds": 30, "memory_mib": 1024}}


@pytest.mark.parametrize("value", [True, 1.0, 0, 2, "1", None])
def test_strict_version(value):
    p = payload()
    p["version"] = value
    with pytest.raises(ValueError):
        prepare(p)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "30", 1e9])
def test_invalid_budget(value):
    for key in ["seconds", "memory_mib"]:
        p = payload()
        p["budget"][key] = value
        with pytest.raises(ValueError):
            prepare(p)


def test_no_payload_paths_or_unreviewed_search_plan():
    for field, value in [("binary", "unused"), ("search_arguments", [2])]:
        p = payload()
        p[field] = value
        with pytest.raises(ValueError):
            prepare(p)


def test_no_implicit_build_download_or_fallback():
    with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match="Provision"):
        reviewed_binary()


def test_prepared_is_defensive_and_wrong_execute_input_rejected():
    p = payload()
    prepared = prepare(p)
    p["task"]["test"][0]["input"][0][0] = 9
    assert prepared.task.test == (((2,),),)
    with pytest.raises(ValueError):
        execute(p)
