"""Otto model-12 native Sofia with embedding and reference-selected interactions.

Historical lambda, iterations, ddof and preprocessing scope remain unresolved.
Scores are nine independent logistic probabilities, without renormalization.
"""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sciona.otto_preprocessing import representation
from sciona.otto_interactions import fit_selection
from sciona.otto_tsne import PopulationEmbedding


def _fit_encoded(reference,labels,query,*,seed,regularization,iterations,r_library):
    x=np.asarray(reference,dtype=float);q=np.asarray(query,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or q.ndim!=2 or not len(x) or not len(q) or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Finite encoded matrices required')
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if type(seed) is not int or not 0<=seed<2**31 or type(iterations) is not int or not 1<=iterations<2**31:
        raise ValueError('Valid integer seed and iterations required')
    if type(regularization) not in (int,float) or not np.isfinite(regularization) or regularization<=0:
        raise ValueError('Finite positive regularization required')
    if type(r_library) is not str or not Path(r_library).is_dir():raise ValueError('Configured R library required')
    with tempfile.TemporaryDirectory(prefix='otto-private-sofia-') as temporary:
        root=Path(temporary)
        for name,value in [('reference',x),('labels',y),('query',q)]:
            np.savetxt(root/name,value,delimiter=',',fmt='%.17g')
        command=['Rscript','--vanilla',str(Path(__file__).with_name('otto_sofia.R')),str(Path(r_library).resolve()),
                 str(root/'reference'),str(root/'labels'),str(root/'query'),str(root/'result'),
                 str(seed),str(regularization),str(iterations)]
        try:
            result=subprocess.run(command,capture_output=True,timeout=60,check=False)
            if result.returncode:raise ValueError('Native Sofia execution failed')
            scores=np.loadtxt(root/'result',delimiter=',',ndmin=2)
        except (OSError,subprocess.TimeoutExpired):
            raise ValueError('Native Sofia runtime unavailable or timed out') from None
    if scores.shape!=(len(q),9) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Invalid native Sofia probabilities')
    return scores


def assemble(reference,labels,reference_ids,query,query_ids,*,embedding,seed,forest,triples,ddof,r_library):
    x=representation(reference,kind='raw');q=representation(query,kind='raw')
    if not isinstance(embedding,PopulationEmbedding):raise ValueError('Fixed population embedding required')
    z=embedding.lookup(reference_ids);v=embedding.lookup(query_ids)
    if len(x)!=len(z) or len(q)!=len(v) or set(reference_ids)&set(query_ids):raise ValueError('Disjoint aligned identities required')
    if type(ddof) is not int or ddof not in (0,1) or len(x)<=ddof:raise ValueError('Explicit valid standardization convention required')
    if type(forest) is not dict or set(forest)!={'ntree','mtry','nodesize'}:raise ValueError('Explicit forest controls required')
    selection=fit_selection(x,labels,seed=seed,r_library=r_library,**forest)
    a=np.column_stack((x,z,selection.transform(x,triples=triples)))
    b=np.column_stack((q,v,selection.transform(q,triples=triples)))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        center=a.mean(axis=0);scale=a.std(axis=0,ddof=ddof)
        scale[scale==0]=1.
        a=(a-center)/scale;b=(b-center)/scale
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(center).all() or not np.isfinite(scale).all():raise ValueError('Nonfinite standardized assembly')
    return a,b


def fit_predict(reference,labels,reference_ids,query,query_ids,*,embedding,seed,controls):
    if type(controls) is not dict or set(controls)!={'forest','triples','ddof','r_library','sofia'}:raise ValueError('Explicit interaction and Sofia controls required')
    options=controls['sofia']
    if type(options) is not dict or set(options)!={'regularization','iterations'}:raise ValueError('Explicit Sofia controls required')
    a,b=assemble(reference,labels,reference_ids,query,query_ids,embedding=embedding,seed=seed,**{k:v for k,v in controls.items() if k!='sofia'})
    return _fit_encoded(a,labels,b,seed=seed,r_library=controls['r_library'],**options)
