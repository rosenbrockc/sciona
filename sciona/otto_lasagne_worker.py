"""Private CPU worker for native Lasagne models; runtime paths are configured."""
from pathlib import Path
import json
import sys
import numpy as np
import theano
import theano.tensor as tensor
import lasagne
from sciona.otto_preprocessing import fit_scaling


def train(x,y,q,*,seed,epochs,controls):
    lasagne.random.set_rng(np.random.RandomState(seed))
    rng=np.random.RandomState(seed)
    xvar=tensor.matrix('features');yvar=tensor.ivector('labels')
    network=lasagne.layers.InputLayer((None,x.shape[1]),input_var=xvar)
    for width in controls['hidden']:
        network=lasagne.layers.DenseLayer(network,width,nonlinearity=lasagne.nonlinearities.rectify)
    network=lasagne.layers.DenseLayer(network,9,nonlinearity=lasagne.nonlinearities.softmax)
    output=lasagne.layers.get_output(network)
    loss=lasagne.objectives.categorical_crossentropy(output,yvar).mean()
    weights=lasagne.layers.get_all_params(network,trainable=True)
    updates=lasagne.updates.nesterov_momentum(loss,weights,learning_rate=controls['learning_rate'],momentum=controls['momentum'])
    step=theano.function([xvar,yvar],loss,updates=updates,mode='FAST_COMPILE')
    predict=theano.function([xvar],lasagne.layers.get_output(network,deterministic=True),mode='FAST_COMPILE')
    for _ in range(epochs):
        order=rng.permutation(len(x))
        for start in range(0,len(x),controls['batch_size']):
            batch=order[start:start+controls['batch_size']]
            value=float(step(x[batch],y[batch]))
            if not np.isfinite(value):raise ValueError('Nonfinite native training loss')
    return predict(q)


def main(root):
    options=json.loads((root/'controls.json').read_text());controls=options['controls']
    with np.load(root/'input.npz',allow_pickle=False) as data:
        x=data['reference'];y=data['labels'].astype('int32');q=data['query']
    kinds=['log1p','raw'] if options['variant']=='pair' else ['log1p']*6
    predictions=[]
    for run,kind in enumerate(kinds):
        scaling=fit_scaling(x,kind=kind,ddof=controls['ddof'])
        a=scaling.transform(x).astype('float32');b=scaling.transform(q).astype('float32')
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid float32 standardized features')
        predictions.append(train(a,y,b,seed=(options['seed']+run)%2**31,epochs=controls['epochs'][run],controls=controls))
    np.save(root/'scores.npy',np.stack(predictions),allow_pickle=False)


if __name__=='__main__':main(Path(sys.argv[1]))
