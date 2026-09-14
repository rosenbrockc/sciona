"""Otto model-11 native Sofia family with reference-fitted raw standardization.

Historical lambda, iterations, ddof and preprocessing scope remain unresolved.
Scores are nine independent logistic probabilities, without renormalization.
"""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sciona.otto_preprocessing import representation, fit_scaling


def fit_predict(reference,labels,query,*,seed,regularization,iterations,ddof,r_library):
    x=representation(reference,kind='raw');q=representation(query,kind='raw');y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if type(seed) is not int or not 0<=seed<2**31 or type(iterations) is not int or not 1<=iterations<2**31:
        raise ValueError('Valid integer seed and iterations required')
    if type(regularization) not in (int,float) or not np.isfinite(regularization) or regularization<=0:
        raise ValueError('Finite positive regularization required')
    if type(r_library) is not str or not Path(r_library).is_dir():raise ValueError('Configured R library required')
    scaler=fit_scaling(x,kind='raw',ddof=ddof)
    x=scaler.transform(x);q=scaler.transform(q)
    with tempfile.TemporaryDirectory(prefix='otto-private-sofia-') as temporary:
        root=Path(temporary)
        for name,value in [('reference',x),('labels',y),('query',q)]:
            np.savetxt(root/name,value,delimiter=',',fmt='%.17g')
        command=['Rscript','--vanilla',str(Path(__file__).with_suffix('.R')),str(Path(r_library).resolve()),
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
