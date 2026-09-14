"""Complete independent Santander neural-to-pseudo-label-to-blend lifecycle."""
from sciona.santander_training import train_neural_cv
from sciona.santander_retraining import retrain


def fit(training,labels,query,*,initial_neural_controls,neural_controls,tree_controls,
        reference_policy,neural_positive=5000,neural_negative=3000,tree_positive=2700,tree_negative=2000):
    """Use initial neural ranks to select both branches' hard pseudo labels.

Tail counts never shrink automatically. The count-reference policy and branch
controls are explicit independent choices, not historical replay claims.
"""
    initial=train_neural_cv(training,labels,query,**initial_neural_controls)
    result=retrain(training,labels,query,initial['mean_ranks'],reference_policy=reference_policy,
        neural_controls=neural_controls,tree_controls=tree_controls,
        neural_positive=neural_positive,neural_negative=neural_negative,tree_positive=tree_positive,tree_negative=tree_negative)
    result['initial_neural']=initial
    return result
