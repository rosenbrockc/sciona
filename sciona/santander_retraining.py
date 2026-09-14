"""Explicit hard-pseudo-label retraining for both Santander model branches.

The selected query rows remain in the unlabeled count reference as well as
joining the labeled population. This is a required explicit independent choice;
the historical membership convention is unresolved. Each learner regenerates
supervised features from its own expanded labeled population before its folds.
"""
import numpy as np
from sciona.santander_pseudo_labels import select_pseudo_labels
from sciona.santander_training import train_neural_cv
from sciona.santander_tree import train_tree_cv
from sciona.santander_blend import blend_neural_tree


def expanded_population(training,labels,query,scores,*,positives,negatives,reference_policy):
    if reference_policy!='retain_selected':raise ValueError('Explicit retain-selected count-reference policy required')
    x,y,q,s=map(np.asarray,(training,labels,query,scores))
    if any(v.ndim!=2 or not all(v.shape) or v.dtype.kind!='f' or not np.isfinite(v).all() for v in (x,q)) or x.shape[1]!=q.shape[1]:
        raise ValueError('Aligned finite floating populations required')
    if y.shape!=(len(x),) or y.dtype.kind not in 'biu' or set(y.tolist())!={0,1}:
        raise ValueError('Aligned binary labeled population required')
    if s.shape!=(len(q),):raise ValueError('Scores must align with the entire query population')
    positions,targets=select_pseudo_labels(s,positives=positives,negatives=negatives)
    return dict(training=np.concatenate((x,q[positions])),labels=np.concatenate((y,targets)),
        query=q.copy(),selected_positions=positions,selected_labels=targets)


def retrain(training,labels,query,neural_scores,*,reference_policy,neural_controls,tree_controls,
            neural_positive=5000,neural_negative=3000,tree_positive=2700,tree_negative=2000):
    # Check both selection boundaries before either costly training lifecycle.
    selections={name:expanded_population(training,labels,query,neural_scores,
        positives=positive,negatives=negative,reference_policy=reference_policy)
        for name,positive,negative in [('neural',neural_positive,neural_negative),('tree',tree_positive,tree_negative)]}
    if type(neural_controls) is not dict or type(tree_controls) is not dict:
        raise ValueError('Explicit branch training controls required')
    result={}
    for name,learner,controls in [('neural',train_neural_cv,neural_controls),('tree',train_tree_cv,tree_controls)]:
        p=selections[name]
        result[name]=learner(p['training'],p['labels'],p['query'],**controls)
    scores=blend_neural_tree(result['neural']['mean_ranks'],result['tree']['mean_ranks'])
    return dict(scores=scores,branches=result,
        selection_counts={name:dict(positive=int(np.sum(p['selected_labels']==1)),negative=int(np.sum(p['selected_labels']==0)),
            labeled_rows=len(p['labels']),query_reference_rows=len(p['query'])) for name,p in selections.items()},
        reference_policy=reference_policy,
        validation_scope='Pseudo-labeled query rows remain in the count reference; global encoding precedes CV and scores are not independent validation.')
