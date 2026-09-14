import numpy as np
import pytest

from sciona.champs_ensemble import COUPLING_TYPES, MODEL_ORDER, blend_predictions


def inputs():
    return {name: {"case-a": float(i), "case-b": float(i)}
            for i, name in enumerate(MODEL_ORDER)}, {"case-a": "1JHC", "case-b": "3JHN"}


def test_alignment_and_hand_calculated_trim():
    predictions, kinds = inputs()
    # Type 0 selects indices 0,1,2,3,5,6,10,11,12; central five=2,3,5,6,10.
    # Type 7 selects 0,1,3,4,5,6,8,11,12; central five=3,4,5,6,8.
    predictions[MODEL_ORDER[3]] = dict(reversed(list(predictions[MODEL_ORDER[3]].items())))
    assert blend_predictions(predictions, kinds) == {"case-a": 5.2, "case-b": 5.2}


@pytest.mark.parametrize("value", [True, np.bool_(False), float("nan"), float("inf"), "1", 1j, 10**1000])
def test_reject_invalid_even_for_trimmed_prediction(value):
    predictions, kinds = inputs()
    predictions[MODEL_ORDER[0]]["case-a"] = value
    with pytest.raises(ValueError):
        blend_predictions(predictions, kinds)


def test_reject_missing_extra_and_misaligned_records():
    predictions, kinds = inputs()
    for change in [lambda p: p.pop(MODEL_ORDER[0]),
                   lambda p: p.update(unexpected={}),
                   lambda p: p[MODEL_ORDER[0]].pop("case-b"),
                   lambda p: p[MODEL_ORDER[0]].update(unexpected=2.)]:
        candidate = {k: dict(v) for k, v in predictions.items()}
        change(candidate)
        with pytest.raises(ValueError):
            blend_predictions(candidate, kinds)


def test_ties_empty_and_overflow():
    predictions = {m: {str(i): 2. for i in range(8)} for m in MODEL_ORDER}
    kinds = dict(zip(map(str, range(8)), COUPLING_TYPES))
    assert list(blend_predictions(predictions, kinds).values()) == [2.] * 8
    assert blend_predictions({m: {} for m in MODEL_ORDER}, {}) == {}
    with pytest.raises(ValueError, match="overflow"):
        blend_predictions({m: {"case": 1e308} for m in MODEL_ORDER}, {"case": "1JHC"})
