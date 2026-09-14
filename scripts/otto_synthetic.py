"""Synthetic-only full Otto execution fixture; native locations supplied at runtime."""
import numpy as np


def payload(*,r_library,libfm):
    rng = np.random.default_rng(12)
    y = np.tile(np.arange(9), 150)
    f = np.arange(len(y)) % 5
    x = np.column_stack((np.eye(9)[y] * 20, np.ones((len(y), 4)))) + rng.poisson(0.3, size=(len(y), 13))
    q = np.array([x[y == i].mean(axis=0) for i in range(9)])
    ids = [f'r{i}' for i in range(len(x))]
    qids = [f'q{i}' for i in range(len(q))]
    forest = dict(ntree=32, mtry=4, nodesize=1)
    boost = dict(rounds=8, max_depth=3, eta=0.4, subsample=0.8, colsample_bytree=0.8)
    cluster = dict(cluster_counts=[2, 3], ddof=1, n_init=2, max_iter=100)
    neural = dict(hidden=[16], epochs=[20] * 2, batch_size=64, learning_rate=0.1, momentum=0.9, ddof=1)
    c = {1: dict(forest, r_library=r_library), 2: dict(C=1.0, max_iter=2000), 3: dict(n_estimators=32, max_features=4, min_samples_leaf=1), 4: dict(neighbors=8, metric='euclidean', ddof=1), 5: dict(executable=libfm, factors=4, iterations=40, learn_rate=0.05, regularization=[0.0, 0.0, 0.01]), 6: dict(hidden=[16], epochs=30, activation='Rectifier', standardize=True), 7: dict(alpha=1.0), 8: neural, 9: dict(neural, epochs=[20] * 6), 10: dict(cluster_counts=[2, 3], n_init=2, max_iter=100), 11: dict(regularization=0.01, iterations=10000, ddof=1, r_library=r_library), 12: dict(forest=forest, triples=[(0, 1, 2), (3, 4, 5)], ddof=1, r_library=r_library, sofia=dict(regularization=0.01, iterations=10000)), 13: dict(forest=forest, triples=[(0, 1, 2), (3, 4, 5)], log_scope='nonnegative_blocks', r_library=r_library, sofia=dict(regularization=0.01, iterations=10000, rank_probability=0.5)), 14: boost, 15: dict(augmentation=dict(cluster, cluster_counts=list(range(2, 9))), boosting=boost), 21: boost, 22: dict(neighbors=8, metric='euclidean', ddof=None), 23: dict(neighbors=8, metric='euclidean', ddof=None)}
    for i in (16, 17, 18):
        c[i] = dict(clustering=cluster, boosting=boost)
    for i, depth in ((19, 2), (20, 3)):
        c[i] = dict(neural, hidden=[16] * depth, epochs=[20, 21, 22] * 40, representation='log1p')
    for i in range(24, 34):
        c[i] = dict(metric='euclidean')
    candidates={
        'xgboost':[dict(boost,rounds=rounds) for rounds in (4,8)],
        'adaboost_extratrees':[dict(algorithm='SAMME',boost_rounds=4,learning_rate=.7,trees=3,max_depth=depth,min_samples_leaf=2,max_features=9) for depth in (1,3)],
        'lasagne_neural':[dict(hidden=[16],epochs=epochs,batch_size=64,learning_rate=.1,momentum=.9,ddof=1) for epochs in (1,20)]}
    return dict(version=1,
        training=dict(values=x.tolist(),labels=y.tolist(),identities=ids,first_folds=f.tolist(),meta_folds=(np.arange(len(y))%4).tolist()),
        query=dict(values=q.tolist(),identities=qids),
        controls=dict(seed=12,embedding=dict(perplexity=20,learning_rate=50,max_iter=350),
            first_level={str(k):v for k,v in c.items()},meta_candidates=candidates,tsne_interpretation='five_features',
            supplemental=dict(raw_metrics=['euclidean'],tfidf_metrics=['euclidean'],embedding_metrics=['euclidean'],
                tfidf_controls=dict(smooth_idf=True,sublinear_tf=False,norm='l2'),cluster_controls=dict(cluster,kind='raw',standardize=False))))
