"""Synthetic boundary and source ranking checks; no competition task data."""
import copy

import pytest

from sciona.arc_contract import merge_answers, parse_answers, validate_task


def task():
    return {"train": [{"input": [[1, 0]], "output": [[1], [0]]}],
            "test": [{"input": [[2, 0]]}, {"input": [[0, 3]]}]}


def test_isolation_and_defensive_copy():
    raw = task()
    validated = validate_task(raw)
    raw["train"][0]["input"][0][0] = 9
    assert validated.source_task(0) == {"train": [{"input": [[1, 0]], "output": [[1], [0]]}],
                                        "test": [{"input": [[2, 0]]}]}
    second = validated.source_task(1)
    second["test"][0]["input"][0][0] = 9
    assert validated.source_task(1)["test"] == [{"input": [[0, 3]]}]
    for index in [-1, 2, True, 0.0]:
        with pytest.raises(ValueError):
            validated.source_task(index)


@pytest.mark.parametrize("grid", [[], [[]], [[0], [0, 1]], [[True]], [[1.0]], [[-1]], [[10]],
                                  [[float("nan")]], [[float("inf")]], [["1"]], [[0] * 31], [[0]] * 31, ((1,),)])
def test_invalid_grid_rejected(grid):
    raw = task()
    raw["train"][0]["input"] = grid
    with pytest.raises(ValueError):
        validate_task(raw)


def test_shape_changes_allowed_and_max_grid_accepted():
    raw = task()
    raw["test"][0]["input"] = [[9] * 30 for _ in range(30)]
    assert len(validate_task(raw).test[0]) == 30


def test_labels_unknown_fields_and_count_limits_rejected():
    invalid = []
    for key in ["train", "test"]:
        for values in [[], [task()[key][0]] * 21, None]:
            raw = task()
            raw[key] = values
            invalid.append(raw)
    raw = task()
    raw["test"][0]["output"] = [[0]]
    invalid.append(raw)
    raw = task()
    raw["path"] = "unused"
    invalid.append(raw)
    raw = task()
    del raw["train"][0]["output"]
    invalid.append(raw)
    for raw in invalid:
        with pytest.raises(ValueError):
            validate_task(copy.deepcopy(raw))


def test_merge_uses_source_lexicographic_tie_break_and_best_duplicate():
    a = parse_answers("00000000_0\n|1| 2\n|2| 2\n|3| 1\n")
    b = parse_answers("00000000_0\n|1| 3\n|4| 2\n|5| -1\n")
    result = merge_answers([a, b])
    assert [(c.serialized, c.score) for c in result] == [("|1|", 3), ("|4|", 2), ("|2|", 2)]


@pytest.mark.parametrize("text", ["", "00000000_0\n", "wrong_0\n|0| 1", "00000000_0\n|1| nan",
                                  "00000000_0\n|1| inf", "00000000_0\n|1| -inf", "00000000_0\n|12|3| 1",
                                  "00000000_0\n|a| 1", "00000000_0\n|0| 1 extra",
                                  "00000000_0\n" + "|0| 1\n" * 4])
def test_corrupt_output_rejected(text):
    with pytest.raises(ValueError):
        parse_answers(text)


def test_source_fallback_is_distinct_from_missing_runs():
    assert merge_answers([parse_answers("00000000_0\n|0| -1\n")])[0].grid == ((0,),)
    assert merge_answers([()])[0].serialized == "|0|"
    with pytest.raises(ValueError):
        merge_answers([])
