import copy
import json
from pathlib import Path

import pytest

from scripts.audit_first_wave_fixture import FIXTURE, classify


def fixture():
    return json.loads((Path(__file__).resolve().parents[2] / FIXTURE).read_text())


def test_exact_source_is_classified_as_synthetic():
    source = fixture()
    assert classify(copy.deepcopy(source), source) == 'synthetic_parser_fixture'


@pytest.mark.parametrize('section', ['equations', 'inference_edges', 'description'])
def test_same_fixture_id_does_not_hide_changed_source(section):
    source = fixture()
    changed = copy.deepcopy(source)
    changed[section] = [] if isinstance(changed[section], list) else 'Different source'
    with pytest.raises(ValueError, match='differs'):
        classify(changed, source)
