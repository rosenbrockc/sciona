"""Native R randomForest bridge for Otto, with private temporary transport."""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sciona.otto_preprocessing import representation


def fit_predict(reference,labels,query,*,seed,ntree,mtry,nodesize,r_library):
    x=representation(reference,kind='raw');q=representation(query,kind='raw');y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    for v in (ntree,mtry,nodesize):
        if type(v) is not int or not 1<=v<2**31:raise ValueError('Positive R integer controls required')
    if type(seed) is not int or not 0<=seed<2**31 or mtry>x.shape[1]:raise ValueError('Invalid seed or candidate-feature count')
    if type(r_library) is not str or not Path(r_library).is_dir():raise ValueError('Configured R library required')
    with tempfile.TemporaryDirectory(prefix='otto-private-r-') as temporary:
        root=Path(temporary)
        for name,value in [('reference',x),('labels',y),('query',q)]:
            np.savetxt(root/name,value,delimiter=',',fmt='%.17g')
        command=['Rscript','--vanilla',str(Path(__file__).with_suffix('.R')),str(Path(r_library).resolve()),
                 str(root/'reference'),str(root/'labels'),str(root/'query'),str(root/'result'),
                 str(seed),str(ntree),str(mtry),str(nodesize)]
        try:
            result=subprocess.run(command,capture_output=True,timeout=60,check=False)
            if result.returncode:raise ValueError('Native R forest execution failed')
            scores=np.loadtxt(root/'result',delimiter=',',ndmin=2)
        except (OSError,subprocess.TimeoutExpired):
            raise ValueError('Native R forest runtime unavailable or timed out') from None
    if scores.shape!=(len(q),9) or not np.isfinite(scores).all() or (scores<0).any() or not np.allclose(scores.sum(axis=1),1.):
        raise ValueError('Invalid native forest probabilities')
    return scores
