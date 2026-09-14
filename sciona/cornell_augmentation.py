"""Pinned Cornell augmentation with private, already-resampled noise banks."""
import ast
import functools
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PINS='f9b89e14127839dba4be4186a0c3c83ae5e4fa2ac376e3271b1949731e7338e9'
DEPENDENCIES='3e87bfc6bec2db33e1b0b01c5c5146f92a4ec2ab7a0e52c60e60b19d5a724617'


def manifest(name,digest):
    raw=(ROOT/'docs/reviews'/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Cornell manifest hash mismatch')
    return json.loads(raw)


def definitions(cache,pins,name,exclude=()):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid software path')
    raw=(Path(cache)/path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pins[name]:raise ValueError('Cornell software hash mismatch')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name not in exclude]
    return compile(ast.Module(body=nodes,type_ignores=[]),name,'exec')


class Augmentation:
    """Source default/secondary recipe; bank waveforms must be mono at 32 kHz.

    Each bank is a nonempty sequence of arrays, without filenames. Randomness
    belongs to this instance. Supplied noise must have at least 0.1 seconds and
    positive finite float32 RMS. No filesystem audio reads or downloads occur.
    """
    def __init__(self,source,dependencies,*,variant,background,short_noises=(),seed=0):
        if variant not in ('default','background'):raise ValueError('Unknown augmentation variant')
        self.python_rng=random.Random(seed);self.numpy_rng=np.random.RandomState(seed)
        banks={}
        for name,values in [('background',background),('short',short_noises)]:
            copied=[]
            for raw in values:
                value=np.asarray(raw)
                if value.ndim!=1 or value.dtype.kind not in 'fiu' or len(value)<3200:
                    raise ValueError('Noise must be a mono waveform of at least 0.1 seconds')
                with np.errstate(over='ignore',invalid='ignore'):
                    value=np.array(value,dtype=np.float32,copy=True)
                    rms=np.sqrt(np.mean(value*value))
                if not np.isfinite(value).all() or not np.isfinite(rms) or rms<=0:
                    raise ValueError('Noise must have positive finite float32 RMS')
                value.flags.writeable=False;copied.append(value)
            banks[name]=copied
        if not banks['background'] or (variant=='background' and not banks['short']):
            raise ValueError('Required noise bank is empty')
        arrays={f'{name}:{i}':value for name,values in banks.items() for i,value in enumerate(values)}
        def paths(name):return [key for key in arrays if key.startswith(name+':')]
        def load(key,rate):
            if rate!=32000:raise ValueError('Noise bank sample rate must be 32000')
            return arrays[key].copy(),rate
        numpy=SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if k!='random'},random=self.numpy_rng)
        ns=dict(np=numpy,random=self.python_rng,functools=functools,get_file_paths=paths)
        deps=manifest('competition_cornell_historical_dependencies.json',DEPENDENCIES)
        dep=next(p for p in deps if p['wheel'].startswith('audiomentations-'))
        pins={p['software_path']:p['sha256'] for p in dep['pins']}
        for name in ('core/transforms_interface.py','core/composition.py','core/utils.py','augmentations/transforms.py'):
            exec(definitions(dependencies,pins,'audiomentations/'+name,exclude=('get_file_paths',)),ns)
        for name in ('AddBackgroundNoise','AddShortNoises'):
            setattr(ns[name],'_'+name+'__load_sound',staticmethod(load))
        source_pins=manifest('competition_cornell_source_pins.json',PINS)
        pins={p['software_path']:p['sha256'] for p in source_pins['pins']}
        exec(definitions(source,pins,'src/augmentations/sed_'+variant+'_augment.py'),ns)
        kwargs=dict(bckgrd_aug_dir='background')
        if variant=='background':kwargs['secondary_bckgrd_aug_dir']='short'
        self.transforms=ns['get_transforms'](**kwargs)

    def __call__(self,waveform,*,training=True):
        values=np.asarray(waveform)
        if values.ndim!=1 or not len(values) or values.dtype.kind not in 'fiu':
            raise ValueError('Expected nonempty real mono waveform')
        if not np.isfinite(values).all() or type(training) is not bool:
            raise ValueError('Expected finite waveform and boolean training')
        result=self.transforms['train' if training else 'valid'](values.copy(),32000)
        if result.shape!=values.shape or not np.isfinite(result).all():
            raise ValueError('Augmentation produced invalid waveform')
        return result
