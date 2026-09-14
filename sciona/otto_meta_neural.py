"""600-run native Lasagne meta ensemble; explicit reference standardization."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import numpy as np
from sciona.otto_preprocessing import representation


def fit_bag(reference,labels,query,*,seed,controls):
    x=np.asarray(reference,dtype=float);q=np.asarray(query,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or q.ndim!=2 or not len(x) or not len(q) or x.shape[1]<1 or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Finite meta matrices required')
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Valid integer seed required')
    runs=600
    required={'hidden','epochs','batch_size','learning_rate','momentum','ddof'}
    if type(controls) is not dict or set(controls)!=required:raise ValueError('Explicit Lasagne controls required')
    if type(controls['hidden']) is not list or not controls['hidden'] or any(type(n) is not int or n<1 for n in controls['hidden']):raise ValueError('Positive hidden widths required')
    if type(controls['epochs']) is not int or controls['epochs']<1:raise ValueError('Positive fixed epoch count required')
    if type(controls['batch_size']) is not int or controls['batch_size']<1:raise ValueError('Positive batch size required')
    for key in ('learning_rate','momentum'):
        value=controls[key]
        if type(value) not in (int,float) or not np.isfinite(value) or not 0<value<1:raise ValueError('Learning rate and momentum must be in (0,1)')
    if type(controls['ddof']) is not int or controls['ddof'] not in (0,1):raise ValueError('Explicit standardization convention required')
    with tempfile.TemporaryDirectory(prefix='otto-private-lasagne-') as temporary:
        root=Path(temporary)
        np.savez(root/'input.npz',reference=x,labels=y,query=q)
        (root/'controls.json').write_text(json.dumps(dict(seed=seed,controls=controls)))
        env=os.environ.copy();env['MPLCONFIGDIR']=str(root/'mpl')
        env['OMP_NUM_THREADS']='1'
        env['THEANO_FLAGS']='device=cpu,floatX=float32,cxx=,blas.ldflags=,base_compiledir='+str(root/'compiled')
        process=None
        try:
            process=subprocess.Popen([sys.executable,'-m','sciona.otto_meta_neural_worker',str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
            process.communicate(timeout=300)
            if process.returncode:raise ValueError('Native Lasagne worker failed')
            scores=np.load(root/'scores.npy',allow_pickle=False)
        except (OSError,subprocess.TimeoutExpired):raise ValueError('Native Lasagne runtime unavailable or timed out') from None
        finally:
            if process is not None:
                # All descendants belong to this worker's newly created group.
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
    if scores.shape!=(runs,len(q),9) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any() or not np.allclose(scores.sum(axis=2),1.,atol=1e-6):raise ValueError('Invalid Lasagne ensemble probabilities')
    scores=scores.astype(np.float64)
    scores/=scores.sum(axis=2,keepdims=True)
    return scores
