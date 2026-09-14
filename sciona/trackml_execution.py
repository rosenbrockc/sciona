"""In-memory execution boundary for the source-grounded TrackML lifecycle.

Runtime tables remain caller-owned. No file loading, diagnostic export or scoring.
"""
import numpy as np
import pandas as pd
from sciona.trackml_defaults import DEFAULT_PARAMS
from sciona.trackml_detector import discover_detector
from sciona.trackml_intersections import Intersector
from sciona.trackml_lifecycle import TrackLifecycle


class TrackMLInputError(ValueError):
    """Runtime input violates a supported source execution precondition."""


def _table(frame, columns):
    if not isinstance(frame,pd.DataFrame) or frame.columns.duplicated().any() or not set(columns)<=set(frame.columns):
        raise TrackMLInputError('Required runtime columns must exist and be unique')
    for column in columns:
        values=frame[column].to_numpy()
        if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
            raise TrackMLInputError('Runtime numeric fields must be finite')


class RuntimeEvent:
    """Source event indexing over explicit runtime tables; no filesystem access."""
    has_truth=False

    def __init__(self,hits,cells,event_id):
        _table(hits,['hit_id','volume_id','layer_id','module_id','x','y','z'])
        ids=hits.hit_id.to_numpy()
        if ids.dtype.kind not in 'iu' or not np.array_equal(np.sort(ids),np.arange(1,len(hits)+1)):
            raise TrackMLInputError('Hit IDs must be unique contiguous integers starting at one')
        for name in ['module_id','layer_id','volume_id']:
            values=hits[name].to_numpy()
            if values.dtype.kind not in 'iu' or (values<0).any():
                raise TrackMLInputError('Detector identifiers must be nonnegative integers')
        if (hits.module_id>np.iinfo(np.int16).max).any():
            raise TrackMLInputError('Module IDs exceed source storage capacity')
        if not isinstance(event_id,(int,np.integer)) or isinstance(event_id,bool) or not 0<=event_id<=32767:
            raise TrackMLInputError('Event identifier must fit the source nonnegative int16 representation')
        self.hits_df=hits.copy(deep=True);self.cells_df=None if cells is None else cells.copy(deep=True)
        self.has_cells=cells is not None;self.event_id=int(event_id);self.max_hit_id=len(hits)
        if cells is not None:
            _table(cells,['hit_id','ch0','ch1','value'])
            if cells.hit_id.dtype.kind not in 'iu' or not cells.hit_id.isin(ids).all() or (cells.value<=0).any():
                raise TrackMLInputError('Cells require valid hit references and positive weights')
            if not np.array_equal(np.sort(cells.hit_id.unique()),np.sort(ids)):
                raise TrackMLInputError('Cell-enabled execution requires cells for every hit')
        self._xyz=np.full((len(hits)+1,3),np.nan,dtype=np.float64)
        self._xyz[ids]=hits[['x','y','z']].to_numpy()
        self._module=np.full(len(hits)+1,-1,dtype=np.int16);self._module[ids]=hits.module_id
        self._isopen=False

    def open(self):self._isopen=True
    def close(self):self._isopen=False
    def summary(self):return 'runtime event'
    def hitCoordinatesById(self,ids):
        if not self._isopen:raise RuntimeError('Event is closed')
        return tuple(self._xyz[np.asarray(ids)].T.copy())
    def hitModuleIdById(self,ids):
        if not self._isopen:raise RuntimeError('Event is closed')
        return self._module[np.asarray(ids)].copy()


def check_layer_population(spec,hits,minimum):
    """Prevent source append-based layer lists from silently changing layer IDs."""
    recognized=0
    for table,key,count in [(spec.cylinders.cylinders_df,'cyl_id',len(spec.cylinders.cyl_rsqr)),
                            (spec.caps.cap_layers_df,'cap_id',len(spec.caps.cap_z))]:
        joined=hits.join(table,on=['volume_id','layer_id'],how='inner')
        recognized+=len(joined)
        sizes=joined.groupby(key).size().reindex(range(count),fill_value=0)
        if (sizes<minimum).any():
            raise TrackMLInputError('Every detector layer must retain enough hits for the configured neighbor search')
    if recognized!=len(hits):raise TrackMLInputError('Hits must resolve to exactly one known detector layer')


class CheckedLifecycle(TrackLifecycle):
    def setupRun(self,event,**kwargs):
        params=self.paramsForRun(kwargs.get('i_commit',0),kwargs.get('sunset',False))
        minimum=max(4,int(params['follow__pairs_k']),int(params['follow__weird_k']) if params['follow__weird_triples'] else 0)
        hits=event.hits_df;used=kwargs.get('used')
        if used is not None:hits=hits.loc[~used[hits.hit_id.to_numpy()]]
        check_layer_population(self.spec,hits,minimum)
        return super().setupRun(event,**kwargs)


def execute_tracking(modules,hits,*,cells=None,layer_functions=None,params=None,event_id=0):
    """Run the full tracking/commit loop and return one assignment per runtime hit.

    Sparse or exhausted layers raise TrackMLInputError instead of remapping layers
    or padding kNN results with invented hits. Complete-layer source behavior is
    preserved. Caller-supplied map objects implement the source getGridValues API.
    """
    event=RuntimeEvent(hits,cells,event_id)
    settings=DEFAULT_PARAMS.copy();settings.update({} if params is None else params)
    if settings['follow__niter']<2:raise TrackMLInputError('Ranking interpolation requires at least two follow rounds')
    spec=discover_detector(modules)
    algo=CheckedLifecycle(spec,Intersector(spec),settings,layer_functions=layer_functions)
    outputs=[]
    algo.createOrAppendSubmissionFile=lambda destination,frame,append=False:outputs.append(frame.copy(deep=True))
    try:
        algo.findTracks(None,[event],submission_filename='runtime',analysis=False,score_intermediate=False,score_final=False)
    finally:event.close()
    if len(outputs)!=1:raise RuntimeError('Tracking did not produce exactly one event result')
    result=outputs[0]
    if not result.hit_id.is_unique or not np.array_equal(np.sort(result.hit_id),np.sort(hits.hit_id)):
        raise RuntimeError('Tracking result violates complete unique hit assignment')
    return result
