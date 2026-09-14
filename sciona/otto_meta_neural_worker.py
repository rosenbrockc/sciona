"""Isolated 600-run meta network worker; seed alone varies across members."""
from pathlib import Path
import json
import sys
import numpy as np
from sciona.otto_lasagne_worker import train


def main(root):
    options=json.loads((root/'controls.json').read_text());controls=options['controls']
    with np.load(root/'input.npz',allow_pickle=False) as data:
        x=data['reference'];y=data['labels'].astype('int32');q=data['query']
    if len(x)<=controls['ddof']:raise ValueError('Insufficient scaling references')
    with np.errstate(over='ignore',invalid='ignore'):
        center=x.mean(axis=0);scale=x.std(axis=0,ddof=controls['ddof']);scale[scale==0]=1.
        a=((x-center)/scale).astype('float32');b=((q-center)/scale).astype('float32')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(center).all() or not np.isfinite(scale).all():raise ValueError('Invalid standardized meta features')
    predictions=[]
    for run in range(600):
        predictions.append(train(a,y,b,seed=(options['seed']+run)%2**31,epochs=controls['epochs'],controls=controls))
    np.save(root/'scores.npy',np.stack(predictions),allow_pickle=False)


if __name__=='__main__':main(Path(sys.argv[1]))
