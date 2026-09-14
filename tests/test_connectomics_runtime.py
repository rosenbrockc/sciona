"""Synthetic contract failures; no private records or external source required."""
import numpy as np
import pytest

from sciona.connectomics_runtime import Prepared, execute, prepare
from sciona.connectomics_source import correct_source


def payload():
    return dict(version=1, signals=np.random.default_rng(2).uniform(0,1,(12,4)),
                mode='simple', directivity=False)


@pytest.mark.parametrize('change', [
    {'version':True}, {'version':2}, {'mode':'unknown'}, {'directivity':1},
    {'signals':np.zeros((12,4))}, {'signals':np.ones((3,4))},
    {'signals':[[True]*4]*12}, {'signals':[[1j]*4]*12},
    {'signals':[[float('nan')]*4]*12}, {'signals':[[-1]*4]*12},
    {'signals':[[1e100]*4]*12}, {'signals':np.zeros(12)},
    {'source_path':'untrusted'},
])
def test_invalid_payload(change):
    with pytest.raises(ValueError):
        prepare(payload() | change)


def test_prepared_copies_input():
    raw=payload(); value=prepare(raw)
    assert value.signals.dtype==np.float32 and value.signals.flags.f_contiguous
    assert not value.signals.flags.writeable
    original=value.signals.copy()
    raw['signals'][:]=0
    np.testing.assert_array_equal(value.signals, original)


def test_manually_constructed_prepared_revalidated(monkeypatch):
    monkeypatch.delenv('SCIONA_CONNECTOMICS_SOURCE_DIR', raising=False)
    with pytest.raises(ValueError, match='Constant'):
        execute(Prepared(np.ones((12,4)), 'simple', False))
    with pytest.raises(ValueError, match='Provision'):
        execute(prepare(payload()))
    with pytest.raises(ValueError, match='Prepared'):
        execute({})


def test_source_correction_fails_closed():
    import hashlib
    with pytest.raises(ValueError, match='hash'):
        correct_source(b'changed', '0'*64)
    source=b'unknown source'
    with pytest.raises(ValueError, match='reviewed correction'):
        correct_source(source, hashlib.sha256(source).hexdigest())
