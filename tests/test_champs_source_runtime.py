"""Fail-closed source/config boundaries; full networks are reviewed separately."""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from sciona.champs_source_runtime import ChampsSourceRuntime


def test_manifest_replacement_rejected(monkeypatch):
    monkeypatch.setattr(Path,'read_bytes',lambda _:b'{}')
    with pytest.raises(ValueError,match='manifest hash mismatch'):
        ChampsSourceRuntime(Path('/unused'))


def test_modified_source_rejected_before_execution(monkeypatch):
    import sciona.champs_source_runtime as module
    content=b'raise RuntimeError("must never execute")'
    manifest=json.dumps({'pins':[{'software_path':'synthetic.py',
        'sha256':hashlib.sha256(content).hexdigest(),
        'git_blob_sha1':hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()}]}).encode()
    monkeypatch.setattr(module,'PINS_SHA256',hashlib.sha256(manifest).hexdigest())
    monkeypatch.setattr(Path,'read_bytes',lambda p:manifest if p.name.endswith('_pins.json') else content+b'\n')
    with pytest.raises(ValueError,match='software source hash mismatch'):
        ChampsSourceRuntime(Path('/unused'))


def test_seed_and_variant_validation():
    runtime=object.__new__(ChampsSourceRuntime)
    for seed in (True,-1,2**32,1.5,'1'):
        with pytest.raises(ValueError,match='seed'):
            runtime.create_model('model_H',seed=seed)
    with pytest.raises(ValueError,match='variant'):
        runtime.model_config('../other')


def test_configuration_return_is_independent():
    runtime=object.__new__(ChampsSourceRuntime)
    runtime.sources=MappingProxyType({'models/model_H/config':b"{'d_model':700,'name':'synthetic'}",
        'config/models.json':b'{"num_atom_types":[5,13,27]}'})
    first=runtime.model_config('model_H')
    first['num_atom_types'][0]=999
    assert runtime.model_config('model_H')=={'dim':700,'num_atom_types':[5,13,27]}
