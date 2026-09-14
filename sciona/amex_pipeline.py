"""Independent full Amex learner orchestration from denoised, mapped sequences.

The explicit input contract starts after schema-specific category mapping and
numeric denoising. Raw preprocessing is supplied by a separate boundary stage.
"""
import numpy as np
from sciona.amex_manual import build
from sciona.amex_sequence import encode_series,prediction_slots
from sciona.amex_assembly import assemble
from sciona.amex_row_model import fit as row_fit
from sciona.amex_tree import fit as tree_fit
from sciona.amex_neural_training import fit as neural_fit


def blend(manual,with_slots,series,combined):
    values=[np.asarray(p) for p in (manual,with_slots,series,combined)]
    if any(p.ndim!=1 or not len(p) or p.shape!=values[0].shape or p.dtype.kind not in 'iuf' or not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in values):raise ValueError('Four aligned probability vectors required')
    a,b,c,d=[p.astype(np.float64) for p in values]
    return a*.3+b*.35+c*.15+d*.1


def fit(training,query,labels,*,zero_fill_columns,seed=42,progress=None):
    for population in (training,query):
        if type(population) is not dict or set(population)!= {'numerics','categories','times','months'}:raise ValueError('Explicit sequence populations required')
    joint={key:training[key]+query[key] for key in training};n=len(training['numerics'])
    blocks=build(**joint,zero_fill_columns=zero_fill_columns)
    encoded=encode_series(joint['numerics'],joint['categories'])['sequences']
    # Row source learner sees mapped numeric category codes, not one-hot series.
    rows=[np.concatenate((x,c),axis=1) for x,c in zip(joint['numerics'],joint['categories'])]
    row=row_fit(rows[:n],labels,rows[n:],seed=seed)
    if progress:progress('row_models_complete')
    row_predictions=row['training_predictions']+row['query_predictions'];assembled=assemble(blocks,row_predictions)
    manual=assembled['tree_features'][:,:-13];slots=prediction_slots(row_predictions)
    tree=tree_fit(manual[:n],slots[:n],labels,manual[n:],slots[n:],seed=seed)
    if progress:progress('downstream_trees_complete')
    features=assembled['neural_features']
    neural=neural_fit(encoded[:n],features[:n],labels,encoded[n:],features[n:],seed=seed)
    if progress:progress('neural_models_complete')
    branches=[tree['variants']['manual']['query_predictions'],tree['variants']['manual_with_slots']['query_predictions'],neural['variants']['series']['query_predictions'],neural['variants']['combined']['query_predictions']]
    return dict(scores=blend(*branches),branch_probabilities=np.stack(branches),models=25,
        model_counts=dict(row=5,downstream_tree=10,neural=10),manual_width=manual.shape[1],sequence_width=encoded[0].shape[1],
        score_kind='literal_weighted_sum_0.9',scope='Independent full CPU learner pipeline; joint population features and source literal nonnormalized blend. Historical accuracy and native parity unqualified.')
