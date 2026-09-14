"""Synthetic-only local H2O runtime check; no catalog publication."""
import tempfile
import numpy as np
import h2o
from h2o.backend import H2OLocalServer
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


def main():
    labels=np.tile(np.arange(9),60)
    x=np.sqrt(np.eye(9)[labels]*20+3/8)
    q=np.sqrt(np.eye(9)*20+3/8)
    with tempfile.TemporaryDirectory(prefix='otto-private-h2o-') as root:
        with H2OLocalServer.start(nthreads=1,max_mem_size=1024**3,ice_root=root,log_dir=root,log_level='ERRR',bind_to_localhost=True,verbose=False) as server:
            process=server._process
            h2o.connect(server=server,verbose=False)
            h2o.no_progress()
            try:
                names=[f'x{i}' for i in range(9)]
                train=h2o.H2OFrame(np.column_stack((x,labels)),column_names=names+['target'])
                train['target']=train['target'].asfactor()
                query=h2o.H2OFrame(q,column_names=names)
                predictions=[]
                for seed in range(10):
                    model=H2ODeepLearningEstimator(hidden=[16],activation='Rectifier',epochs=30,seed=seed,reproducible=True,stopping_rounds=0,standardize=True,train_samples_per_iteration=-1,adaptive_rate=True)
                    model.train(x=names,y='target',training_frame=train)
                    predicted=model.predict(query)
                    scores=np.asarray(predicted.as_data_frame(use_pandas=False)[1:],dtype=float)[:,1:]
                    assert scores.shape==(9,9) and np.isfinite(scores).all()
                    np.testing.assert_allclose(scores.sum(axis=1),1.,atol=1e-6)
                    predictions.append(scores)
                mean=np.mean(predictions,axis=0)
                np.testing.assert_array_equal(mean.argmax(axis=1),np.arange(9))
            finally:
                h2o.connection().close()
        process.wait(timeout=10)
        assert process.poll() is not None
    print('PASS: ten native H2O neural fits, nine-class ensemble learning, normalized probabilities, server terminated')


if __name__=='__main__':main()
