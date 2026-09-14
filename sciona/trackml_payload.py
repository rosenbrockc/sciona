"""JSON-compatible runtime tables/maps for the source-grounded tracking executor.

This module never persists payloads. Runtime dataset contents remain caller-owned.
"""
import copy
import numpy as np
import pandas as pd
from sciona.trackml_execution import TrackMLInputError, execute_tracking


DTYPES={'int8','int16','int32','int64','uint8','uint16','uint32','uint64','float32','float64'}


def encode_table(frame):
    if not isinstance(frame,pd.DataFrame) or frame.columns.duplicated().any():
        raise TrackMLInputError('A table with unique columns is required')
    columns=[]
    for name in frame.columns:
        a=frame[name].to_numpy()
        if not isinstance(name,str) or str(a.dtype) not in DTYPES or not np.isfinite(a).all():
            raise TrackMLInputError('Tables require named finite numeric columns')
        columns.append({'name':name,'dtype':str(a.dtype),'values':a.tolist()})
    return {'columns':columns}


def decode_table(payload):
    if not isinstance(payload,dict) or set(payload)!={'columns'} or not isinstance(payload['columns'],list):
        raise TrackMLInputError('Invalid table payload')
    columns={};size=None
    for column in payload['columns']:
        if not isinstance(column,dict) or set(column)!={'name','dtype','values'}:
            raise TrackMLInputError('Invalid column payload')
        name=column['name'];dtype=column['dtype'];values=column['values']
        if not isinstance(name,str) or name in columns or dtype not in DTYPES or not isinstance(values,list):
            raise TrackMLInputError('Invalid column specification')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values):
            raise TrackMLInputError('Column values must be numeric scalars')
        if size is None:size=len(values)
        if len(values)!=size:raise TrackMLInputError('Column lengths must agree')
        target=np.dtype(dtype)
        if target.kind in 'iu':
            bounds=np.iinfo(target)
            if any(not np.isfinite(v) or v!=int(v) or not bounds.min<=v<=bounds.max for v in values):
                raise TrackMLInputError('Integer column values exceed their representation')
        with np.errstate(over='ignore',invalid='ignore'):a=np.asarray(values,dtype=target)
        if not np.isfinite(a).all():raise TrackMLInputError('Column values must remain finite')
        columns[name]=a
    return pd.DataFrame(columns)


class GridLayerFunctions:
    """Explicit source grid API: omitted grids are unavailable; null values are NaN."""
    def __init__(self,payload):
        if not isinstance(payload,dict) or set(payload)!={'functions','grids'}:
            raise TrackMLInputError('Invalid layer-function payload')
        names=payload['functions']
        if not isinstance(names,list) or not all(isinstance(n,str) and n for n in names) or len(set(names))!=len(names):
            raise TrackMLInputError('Layer function names must be unique strings')
        if not isinstance(payload['grids'],list):raise TrackMLInputError('Layer grids must be a list')
        self.functions=tuple(names);self._grids={}
        for grid in payload['grids']:
            if not isinstance(grid,dict) or set(grid)!={'is_cylinder','layer_id','function','axes','values'}:
                raise TrackMLInputError('Invalid layer grid')
            kind=grid['is_cylinder'];layer=grid['layer_id'];name=grid['function']
            if type(kind) is not bool or type(layer) is not int or layer<0 or name not in self.functions:
                raise TrackMLInputError('Invalid grid identity')
            key=(kind,layer,name)
            if key in self._grids:raise TrackMLInputError('Duplicate layer grid')
            axes=[np.asarray(axis,dtype=np.float64) for axis in grid['axes']]
            if not 1<=len(axes)<=3 or any(a.ndim!=1 or not len(a) or not np.isfinite(a).all() or not (np.diff(a)>0).all() for a in axes):
                raise TrackMLInputError('Grid axes must be finite and strictly increasing')
            active=False
            for axis in axes:
                if len(axis)>1:active=True
                elif active:raise TrackMLInputError('Source grids only support leading singleton axes')
            if not active:raise TrackMLInputError('A grid requires a nonsingleton axis')
            values=np.asarray(grid['values'],dtype=np.float64)
            if values.shape!=tuple(len(a) for a in axes) or np.isinf(values).any():
                raise TrackMLInputError('Grid values must match axes and exclude infinity')
            self._grids[key]=(tuple(a.copy() for a in axes),values.copy())

    def getGridValues(self,*,is_cylinder,layer_id,functions):
        result=[]
        for name in functions:
            if name not in self.functions:raise TrackMLInputError('Unknown requested layer function')
            item=self._grids.get((is_cylinder,layer_id,name))
            result.append((None,None) if item is None else (tuple(a.copy() for a in item[0]),item[1].copy()))
        return result


def encode_layer_functions(functions,*,cylinder_count,cap_count):
    grids=[]
    for kind,count in [(True,cylinder_count),(False,cap_count)]:
        for layer in range(count):
            data=functions.getGridValues(is_cylinder=kind,layer_id=layer,functions=functions.functions)
            if len(data)!=len(functions.functions):raise TrackMLInputError('Missing function results')
            for name,(axes,values) in zip(functions.functions,data):
                if axes is None:continue
                a=np.asarray(values)
                grids.append(dict(is_cylinder=kind,layer_id=layer,function=name,
                    axes=[np.asarray(axis).tolist() for axis in axes],
                    values=np.where(np.isnan(a),None,a).tolist()))
    payload={'functions':list(functions.functions),'grids':grids}
    GridLayerFunctions(payload)
    return payload


def decode_tracking_payload(payload):
    if not isinstance(payload,dict) or set(payload)-{'version','modules','hits','cells','layer_functions','params','event_id'}:
        raise TrackMLInputError('Invalid tracking payload')
    if payload.get('version')!=1:raise TrackMLInputError('Unsupported tracking payload version')
    maps=payload.get('layer_functions');cells=payload.get('cells')
    return dict(modules=decode_table(payload['modules']),hits=decode_table(payload['hits']),
        cells=None if cells is None else decode_table(cells),
        layer_functions=None if maps is None else GridLayerFunctions(maps),
        params=copy.deepcopy(payload.get('params')),event_id=payload.get('event_id',0))


def run_decoded_tracking(prepared):
    return execute_tracking(**prepared)


def encode_tracking_result(assignments):
    return {'version':1,'assignments':encode_table(assignments)}


def execute_payload(payload):
    return encode_tracking_result(run_decoded_tracking(decode_tracking_payload(payload)))
