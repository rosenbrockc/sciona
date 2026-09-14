"""Native libFM SGD component; historical multiclass/optimizer choices unresolved.

Each (column, observed value) is a distinct categorical level. Fit vocabulary
on references only; ignore unseen query levels. Nine independent binary models
return unnormalized OVR probabilities. These choices are explicit realizations.
"""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.sparse import csr_matrix
from sciona.otto_preprocessing import representation


def encode(reference, query):
    x=representation(reference,kind='raw');q=representation(query,kind='raw')
    if x.shape[1]!=q.shape[1]:raise ValueError('Aligned feature widths required')
    vocabulary=[];offset=0
    for column in x.T:
        levels=np.unique(column)
        vocabulary.append({value:offset+i for i,value in enumerate(levels)})
        offset+=len(levels)
    def transform(values):
        rows=[];cols=[]
        for row,values_row in enumerate(values):
            for mapping,value in zip(vocabulary,values_row):
                if value in mapping:rows.append(row);cols.append(mapping[value])
        return csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(values),offset))
    return transform(x),transform(q)


def _write(path, matrix, labels):
    with path.open('w') as handle:
        for row,label in enumerate(labels):
            indices=matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]]
            entries=[f'{i}:1' for i in indices]
            # Keep dimension independent of query vocabulary, including empty rows.
            if not len(indices) or indices[-1]!=matrix.shape[1]-1:
                entries.append(f'{matrix.shape[1]-1}:0')
            handle.write(str(int(label))+' '+' '.join(entries)+'\n')


def fit_predict(reference, labels, query, *, executable, seed, factors, iterations, learn_rate, regularization):
    x,q=encode(reference,query);y=np.asarray(labels)
    if y.shape!=(x.shape[0],) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class integer labels required')
    if type(seed) is not int or not 0<=seed<2**31:
        raise ValueError('Nonnegative signed integer seed required')
    if any(type(v) is not int or not 1<=v<2**31 for v in (factors,iterations)):
        raise ValueError('Positive factor and iteration counts required')
    if type(learn_rate) not in (int,float) or not np.isfinite(learn_rate) or not 0<learn_rate<=1:
        raise ValueError('Learning rate in (0,1] required')
    if type(regularization) is not list or len(regularization)!=3 or any(type(v) not in (int,float) or not np.isfinite(v) or v<0 for v in regularization):
        raise ValueError('Three finite nonnegative regularizers required')
    if type(executable) is not str or not Path(executable).is_file():
        raise ValueError('Configured native libFM executable required')
    binary=str(Path(executable).resolve());scores=np.empty((q.shape[0],9))
    with tempfile.TemporaryDirectory(prefix='otto-private-fm-') as temporary:
        root=Path(temporary);_write(root/'query',q,np.zeros(q.shape[0],dtype=int))
        for label in range(9):
            (root/'result').unlink(missing_ok=True)
            _write(root/'reference',x,np.where(y==label,1,-1))
            command=[binary,'-task','c','-method','sgd','-train',str(root/'reference'),'-test',str(root/'query'),'-out',str(root/'result'),'-seed',str(seed),'-dim',f'1,1,{factors}','-iter',str(iterations),'-learn_rate',str(learn_rate),'-regular',','.join(map(str,regularization)),'-init_stdev','0.1']
            try:
                result=subprocess.run(command,capture_output=True,timeout=60,check=False)
                if result.returncode:raise ValueError('Native libFM fit failed')
                values=np.loadtxt(root/'result',ndmin=1)
                if values.shape!=(q.shape[0],):raise ValueError('Native prediction shape mismatch')
                scores[:,label]=values
            except (OSError,subprocess.TimeoutExpired):
                raise ValueError('Native libFM runtime unavailable or timed out') from None
    if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Invalid native probabilities')
    return scores
