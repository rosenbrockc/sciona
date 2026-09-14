"""Synthetic Lasagne/Theano CPU validation; no historical GPU claim."""
import numpy as np
import theano
import theano.tensor as tensor
import lasagne


def train(hidden):
    lasagne.random.set_rng(np.random.RandomState(12))
    xvar=tensor.matrix('features');yvar=tensor.ivector('labels')
    network=lasagne.layers.InputLayer((None,9),input_var=xvar)
    for width in hidden:
        network=lasagne.layers.DenseLayer(network,width,nonlinearity=lasagne.nonlinearities.rectify)
    network=lasagne.layers.DenseLayer(network,9,nonlinearity=lasagne.nonlinearities.softmax)
    output=lasagne.layers.get_output(network)
    loss=lasagne.objectives.categorical_crossentropy(output,yvar).mean()
    weights=lasagne.layers.get_all_params(network,trainable=True)
    updates=lasagne.updates.nesterov_momentum(loss,weights,learning_rate=.1,momentum=.9)
    step=theano.function([xvar,yvar],loss,updates=updates,mode='FAST_COMPILE')
    predict=theano.function([xvar],lasagne.layers.get_output(network,deterministic=True),mode='FAST_COMPILE')
    y=np.tile(np.arange(9),20).astype('int32');x=np.eye(9,dtype='float32')[y]
    losses=[float(step(x,y)) for _ in range(100)]
    scores=predict(np.eye(9,dtype='float32'))
    assert np.isfinite(losses).all() and losses[-1]<losses[0]/10
    assert scores.shape==(9,9) and np.isfinite(scores).all()
    np.testing.assert_array_equal(scores.argmax(axis=1),np.arange(9))
    np.testing.assert_allclose(scores.sum(axis=1),1.,atol=1e-6)
    np.testing.assert_allclose(scores[:1],predict(np.eye(9,dtype='float32')[:1]),atol=1e-6)
    return scores


if __name__=='__main__':
    for hidden in ([16,16],[16,16,16]):
        np.testing.assert_array_equal(train(hidden),train(hidden))
    print('PASS: two- and three-hidden-layer Lasagne networks train, repeat and predict nine classes on CPU')
