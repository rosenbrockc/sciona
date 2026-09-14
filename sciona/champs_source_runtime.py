"""Load the reviewed Bosch CHAMPS networks from a pinned software-only cache.

MIT source attribution: docs/licenses/CHAMPS-MIT.txt. This corrected execution
version uses explicit triplet-layout fixes; original pretrained parity is not
claimed. No upstream launchers, data readers, or pickled models are executed.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from sciona.champs_ensemble import MODEL_ORDER
from sciona.champs_source_corrections import correct_triplet_layout

PINS_SHA256 = '99fde10617f64e20fc0459e73723ded08e3692518afd6e74bb67b8f96de804ca'
EXECUTION_VERSION = 'champs-corrected-source.v1'


class ChampsSourceRuntime:
    """An immutable verified source snapshot with isolated model namespaces."""

    def __init__(self, cache: Path):
        manifest = (Path(__file__).resolve().parents[1] /
                    'docs/reviews/competition_champs_source_pins.json').read_bytes()
        if hashlib.sha256(manifest).hexdigest() != PINS_SHA256:
            raise ValueError('CHAMPS source manifest hash mismatch')
        pins = json.loads(manifest)
        sources, hashes = {}, {}
        for entry in pins['pins']:
            path = Path(entry['software_path'])
            if path.is_absolute() or '..' in path.parts or path.as_posix() in sources:
                raise ValueError('Invalid CHAMPS source manifest entry')
            raw = (Path(cache) / path).read_bytes()
            blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if (hashlib.sha256(raw).hexdigest() != entry['sha256'] or
                    blob != entry['git_blob_sha1']):
                raise ValueError('CHAMPS software source hash mismatch')
            sources[path.as_posix()] = raw
            hashes[path.as_posix()] = entry['sha256']
        catalog = json.loads(sources['config/models.json'])
        order = tuple(catalog[name+'_dir'] for name in catalog['names'])
        if order != MODEL_ORDER:
            raise ValueError('CHAMPS model order differs from reviewed ensemble')
        self.sources = MappingProxyType(sources)
        self.hashes = MappingProxyType(hashes)
        self.commit = pins['commit']

    def model_config(self, variant: str) -> dict:
        if variant not in MODEL_ORDER:
            raise ValueError('Unknown CHAMPS model variant')
        config = ast.literal_eval(self.sources[f'models/{variant}/config'].decode())
        config.pop('name', None)
        config['dim'] = config.pop('d_model')
        catalog = json.loads(self.sources['config/models.json'])
        config.update({k:v for k,v in catalog.items() if k.startswith('num_')})
        return config

    def create_model(self, variant: str, *, seed: int = 1729) -> torch.nn.Module:
        """Construct the full configured CPU network, preserving caller RNG state."""
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**32:
            raise ValueError('CHAMPS seed must be an unsigned 32-bit integer')
        config = self.model_config(variant)
        namespace = {'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'np':np,
                     'weight_norm':torch.nn.utils.weight_norm,'colored':lambda text,*_:text}
        folder = f'models/{variant}'
        for relative in ('modules/hierarchical_embedding.py','modules/embeddings.py','graph_transformer.py'):
            path = folder+'/'+relative
            source = self.sources[path]
            if relative == 'graph_transformer.py':
                source = correct_triplet_layout(source,self.hashes[path],kind='graph').encode()
            definitions = [n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
            exec(compile(ast.Module(body=definitions,type_ignores=[]),path,'exec'),namespace)
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                np.random.seed(seed)
                return namespace['GraphTransformer'](**config)
        finally:
            np.random.set_state(numpy_state)
