"""Explicit triple-product subset from thirteen native forest-ranked features.

Gini importance, rank tie order and the selected triples are independent choices;
the source does not identify its exact importance variant or interaction subset.
"""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sciona.otto_preprocessing import representation


@dataclass(frozen=True,repr=False)
class InteractionSelection:
    width: int
    selected: tuple

    def transform(self, values, *, triples):
        x=representation(values,kind='raw')
        if x.shape[1]!=self.width:raise ValueError('Feature width differs from fitted selector')
        if type(triples) is not list or not triples:raise ValueError('Explicit nonempty triple subset required')
        seen=set();columns=[]
        for triple in triples:
            if type(triple) is not tuple or len(triple)!=3 or any(type(i) is not int or not 0<=i<13 for i in triple) or len(set(triple))!=3:
                raise ValueError('Each triple requires three distinct selected-feature ranks')
            key=tuple(sorted(triple))
            if key in seen:raise ValueError('Duplicate interaction triple')
            seen.add(key)
            with np.errstate(over='ignore',invalid='ignore'):
                product=np.prod(x[:,[self.selected[i] for i in triple]],axis=1)
            if not np.isfinite(product).all():raise ValueError('Nonfinite interaction product')
            columns.append(product)
        return np.column_stack(columns)


def fit_selection(reference, labels, *, seed, ntree, mtry, nodesize, r_library):
    x=representation(reference,kind='raw');y=np.asarray(labels)
    if x.shape[1]<13 or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Thirteen or more features and aligned nine-class labels required')
    if type(seed) is not int or not 0<=seed<2**31 or any(type(v) is not int or not 1<=v<2**31 for v in (ntree,mtry,nodesize)) or mtry>x.shape[1]:
        raise ValueError('Valid native forest controls required')
    if type(r_library) is not str or not Path(r_library).is_dir():raise ValueError('Configured R library required')
    with tempfile.TemporaryDirectory(prefix='otto-private-interactions-') as temporary:
        root=Path(temporary)
        np.savetxt(root/'reference',x,delimiter=',',fmt='%.17g');np.savetxt(root/'labels',y,fmt='%d')
        command=['Rscript','--vanilla',str(Path(__file__).with_suffix('.R')),str(Path(r_library).resolve()),str(root/'reference'),str(root/'labels'),str(root/'result'),str(seed),str(ntree),str(mtry),str(nodesize)]
        try:
            result=subprocess.run(command,capture_output=True,timeout=60,check=False)
            if result.returncode:raise ValueError('Native importance execution failed')
            scores=np.loadtxt(root/'result',ndmin=1)
        except (OSError,subprocess.TimeoutExpired):raise ValueError('Native importance runtime unavailable or timed out') from None
    if scores.shape!=(x.shape[1],) or not np.isfinite(scores).all() or (scores<0).any():raise ValueError('Invalid native importance')
    selected=tuple(int(i) for i in np.argsort(-scores,kind='stable')[:13])
    return InteractionSelection(x.shape[1],selected)
