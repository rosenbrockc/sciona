"""Private CPU worker for native Lasagne models; runtime paths are configured."""
from pathlib import Path
import json
import sys
import numpy as np
import theano
import theano.tensor as tensor
import lasagne
from sciona.otto_preprocessing import fit_scaling


from sciona.otto_lasagne_worker import train


def main(root):
    options=json.loads((root/'controls.json').read_text());controls=options['controls']
    with np.load(root/'input.npz',allow_pickle=False) as data:
        x=data['reference'];y=data['labels'].astype('int32');q=data['query']
    kinds=[controls['representation']]*120
    predictions=[]
    for run,kind in enumerate(kinds):
        scaling=fit_scaling(x,kind=kind,ddof=controls['ddof'])
        a=scaling.transform(x).astype('float32');b=scaling.transform(q).astype('float32')
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid float32 standardized features')
        predictions.append(train(a,y,b,seed=(options['seed']+run)%2**31,epochs=controls['epochs'][run],controls=controls))
    np.save(root/'scores.npy',np.stack(predictions),allow_pickle=False)


if __name__=='__main__':main(Path(sys.argv[1]))
