"""Versioned private raw chemistry payload boundary for the CHAMPS pipeline."""
from dataclasses import dataclass
import math

import pandas as pd

from sciona.champs_ensemble import COUPLING_TYPES,MODEL_ORDER
from sciona.champs_training import TrainingOptions


@dataclass
class Prepared:
    training_atoms: pd.DataFrame
    training_couplings: pd.DataFrame
    inference_atoms: pd.DataFrame
    inference_couplings: pd.DataFrame
    inference_keys: dict
    inference_types: dict
    model_options: dict
    selection: str


def _exact(value,fields):
    if type(value) is not dict or set(value)!=set(fields):
        raise ValueError('Exact CHAMPS payload fields required')


def _population(value,labeled):
    if type(value) is not list or not value:
        raise ValueError('CHAMPS population must be a nonempty list')
    atoms,couplings=[],[]
    molecule_keys,coupling_keys=set(),set()
    mapping,types={},{}
    for molecule in value:
        _exact(molecule,('key','elements','coordinates','couplings'))
        key=molecule['key']
        if not isinstance(key,str) or not key or key in molecule_keys:
            raise ValueError('CHAMPS molecule keys must be unique nonempty strings')
        molecule_keys.add(key)
        elements,coordinates=molecule['elements'],molecule['coordinates']
        if (type(elements) is not list or not 3<=len(elements)<=29 or
            any(e not in ('H','C','N','O','F') for e in elements) or
            type(coordinates) is not list or len(coordinates)!=len(elements)):
            raise ValueError('Invalid CHAMPS element/geometry dimensions')
        for i,(element,xyz) in enumerate(zip(elements,coordinates)):
            if type(xyz) is not list or len(xyz)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in xyz):
                raise ValueError('CHAMPS coordinates must be finite real triples')
            atoms.append(dict(molecule_name=key,atom_index=i,atom=element,x=xyz[0],y=xyz[1],z=xyz[2]))
        if type(molecule['couplings']) is not list or not molecule['couplings']:
            raise ValueError('CHAMPS molecule needs requested couplings')
        pairs=set()
        for coupling in molecule['couplings']:
            _exact(coupling,('key','atoms','type','target') if labeled else ('key','atoms','type'))
            ck=coupling['key']
            indices=coupling['atoms']
            if not isinstance(ck,str) or not ck or ck in coupling_keys:
                raise ValueError('CHAMPS coupling keys must be unique nonempty strings')
            if (type(indices) is not list or len(indices)!=2 or any(type(i) is not int or not 0<=i<len(elements) for i in indices) or indices[0]==indices[1]):
                raise ValueError('Invalid CHAMPS coupling atom references')
            pair=tuple(sorted(indices))
            if pair in pairs:
                raise ValueError('Duplicate CHAMPS molecular coupling pair')
            pairs.add(pair)
            if coupling['type'] not in COUPLING_TYPES:
                raise ValueError('Unknown CHAMPS coupling type')
            if 'H' not in (elements[i] for i in indices):
                raise ValueError('CHAMPS source couplings require hydrogen')
            coupling_keys.add(ck)
            identifier=len(couplings)
            if identifier>=2**24:
                raise ValueError('CHAMPS internal ID capacity exceeded')
            row=dict(id=identifier,molecule_name=key,atom_index_0=indices[0],atom_index_1=indices[1],type=coupling['type'])
            if labeled:
                target=coupling['target']
                if type(target) not in (int,float) or not math.isfinite(target):
                    raise ValueError('CHAMPS target must be finite real')
                row['scalar_coupling_constant']=target
            couplings.append(row)
            mapping[identifier]=ck
            types[ck]=coupling['type']
    return pd.DataFrame(atoms),pd.DataFrame(couplings),mapping,types


def prepare(payload):
    _exact(payload,('version','training','inference','models','selection'))
    if type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('Unknown CHAMPS payload version')
    if payload['selection'] not in ('full','validation'):
        raise ValueError('CHAMPS selection must be full or validation')
    models=payload['models']
    _exact(models,MODEL_ORDER)
    options={}
    for variant,value in models.items():
        _exact(value,('epochs','options'))
        if type(value['epochs']) is not int or value['epochs']<=0 or type(value['options']) is not dict:
            raise ValueError('CHAMPS model requires positive epochs and option mapping')
        try: config=TrainingOptions(**value['options'])
        except TypeError: raise ValueError('Unknown CHAMPS training option') from None
        config.validate()
        options[variant]=(value['epochs'],config)
    ta,tc,_,_=_population(payload['training'],True)
    ia,ic,keys,types=_population(payload['inference'],False)
    return Prepared(ta,tc,ia,ic,keys,types,options,payload['selection'])
