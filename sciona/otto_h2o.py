"""Isolated native H2O ten-run bag for Otto's square-root representation."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import numpy as np
from sciona.otto_preprocessing import representation


def fit_predict(reference,labels,query,*,seed,controls):
    x=representation(reference,kind='sqrt_offset');q=representation(query,kind='sqrt_offset');y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Valid integer seed required')
    if type(controls) is not dict or set(controls)!={'hidden','epochs','activation','standardize'}:raise ValueError('Explicit neural controls required')
    if type(controls['hidden']) is not list or not controls['hidden'] or any(type(n) is not int or n<1 for n in controls['hidden']):raise ValueError('Positive hidden layer widths required')
    if type(controls['epochs']) not in (int,float) or not np.isfinite(controls['epochs']) or controls['epochs']<=0:raise ValueError('Positive finite epochs required')
    if controls['activation'] not in ('Rectifier','Tanh') or type(controls['standardize']) is not bool:raise ValueError('Explicit activation and scaling required')
    with tempfile.TemporaryDirectory(prefix='otto-private-h2o-') as temporary:
        root=Path(temporary)
        np.savez(root/'input.npz',reference=x,labels=y,query=q)
        (root/'controls.json').write_text(json.dumps(dict(seed=seed,controls=controls)))
        env=os.environ.copy();env['MPLCONFIGDIR']=str(root/'mpl')
        process=None
        try:
            process=subprocess.Popen([sys.executable,'-m','sciona.otto_h2o_worker',str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
            process.communicate(timeout=300)
            if process.returncode:raise ValueError('Native H2O worker failed')
            scores=np.load(root/'scores.npy',allow_pickle=False)
        except (OSError,subprocess.TimeoutExpired):raise ValueError('Native H2O runtime unavailable or timed out') from None
        finally:
            if process is not None:
                # All descendants belong to this worker's newly created group.
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
    if scores.shape!=(10,len(q),9) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any() or not np.allclose(scores.sum(axis=2),1.,atol=1e-6):raise ValueError('Invalid H2O ensemble probabilities')
    return scores.mean(axis=0)
