"""Private, integrity-bound OpenVaccine model/Adam checkpoints.

These files may contain learned private state and must not be published as
review evidence. Only caller-held manifest digests establish expected identity.
Random streams, input provenance and training schedules are not checkpointed.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from sciona import openvaccine_models as models


def _digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(runtime, directory):
    directory=Path(directory)
    runtime.model.optimizer.build(runtime.model.trainable_variables)
    if any(not np.isfinite(v.numpy()).all() for v in runtime.model.weights+runtime.model.optimizer.variables()):
        raise ValueError('Cannot checkpoint nonfinite state')
    directory.mkdir(parents=True,exist_ok=False)
    checkpoint=runtime.tf.train.Checkpoint(model=runtime.model,optimizer=runtime.model.optimizer)
    checkpoint.save(str(directory/'state'))
    files={p.name:_digest(p) for p in directory.glob('state-1.*')}
    manifest=dict(format_version=1,family=runtime.family,source_sha256=models.SOURCE_SHA256,
                  runtime_sha256=_digest(models.__file__),tensorflow_version=runtime.tf.__version__,
                  optimizer_iteration=int(runtime.model.optimizer.iterations.numpy()),files=files)
    path=directory/'manifest.json'
    path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return _digest(path)


def load_checkpoint(source_dir,directory,*,expected_manifest_sha256,expected_family):
    directory=Path(directory)
    path=directory/'manifest.json'
    if _digest(path)!=expected_manifest_sha256:
        raise ValueError('Checkpoint manifest identity mismatch')
    manifest=json.loads(path.read_text())
    if (manifest.get('format_version')!=1 or manifest.get('family')!=expected_family
            or manifest.get('source_sha256')!=models.SOURCE_SHA256
            or manifest.get('runtime_sha256')!=_digest(models.__file__)):
        raise ValueError('Checkpoint source, runtime or family mismatch')
    files=manifest['files']
    if not isinstance(files,dict) or 'state-1.index' not in files or len(files)<2:
        raise ValueError('Incomplete checkpoint file inventory')
    for name,digest in files.items():
        if name!='state-1.index' and re.fullmatch(r'state-1[.]data-[0-9]{5}-of-[0-9]{5}',name) is None:
            raise ValueError('Invalid checkpoint member name')
        if _digest(directory/name)!=digest:
            raise ValueError('Checkpoint member identity mismatch')
    runtime=models.create_model(source_dir,expected_family)
    if runtime.tf.__version__!=manifest['tensorflow_version']:
        raise ValueError('Checkpoint TensorFlow version mismatch')
    runtime.model.optimizer.build(runtime.model.trainable_variables)
    checkpoint=runtime.tf.train.Checkpoint(model=runtime.model,optimizer=runtime.model.optimizer)
    checkpoint.restore(str(directory/'state-1')).assert_consumed()
    if int(runtime.model.optimizer.iterations.numpy())!=manifest['optimizer_iteration']:
        raise ValueError('Checkpoint optimizer iteration mismatch')
    if any(not np.isfinite(v.numpy()).all() for v in runtime.model.weights+runtime.model.optimizer.variables()):
        raise ValueError('Restored state is nonfinite')
    return runtime
