"""Complete training/prediction composition; catalog graph and metrics pending."""
from sciona.flavours_features import prepare_features, classifier_views
from sciona.flavours_mass_correction import correct_mass
from sciona.flavours_classifiers import train_classifiers
from sciona.flavours_neural import train_neural, neural_partitions
from sciona.flavours_blend import blend
import numpy as np


def execute(training_inputs, labels, mass, query_inputs, excluded_column):
    """Physical inputs are positional tuples matching prepare_features, sans exclusion.

    Caller keeps labeled and query populations disjoint and orders base columns
    consistently with the source. Query control populations can be concatenated
    before this call; their labels/masses are never used for fitting.
    """
    training = prepare_features(*training_inputs, excluded_column)
    query = prepare_features(*query_inputs, excluded_column)
    labels = np.asarray(labels)
    mass = np.asarray(mass, dtype=np.float64)
    expected = (len(training['base']),)
    if labels.shape != expected or mass.shape != expected or not np.isfinite(mass).all():
        raise ValueError('Finite mass and labels must align with the training population')
    neural_partitions(labels)
    corrected = correct_mass(training['regression'], mass, query['regression'])
    tv = classifier_views(training, corrected['oof_mass'])
    qv = classifier_views(query, corrected['query_mass'])
    trees = train_classifiers(tv, labels, qv)
    neural = train_neural(tv['corrected'], labels, qv['corrected'])
    return dict(query_scores=blend(trees['query'], neural['query']),
                oof_scores=blend(trees['oof'], neural['oof']),
                mass_correction=corrected, classifiers=trees, neural=neural)
