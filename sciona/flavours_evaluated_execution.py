"""Train once, predict query/control populations, and apply source diagnostics."""
import numpy as np
from sciona.flavours_features import prepare_features
from sciona.flavours_evaluation import evaluate
from sciona.flavours_pipeline import execute


def execute_evaluated(training_inputs, labels, mass, query_inputs, controls, excluded_column):
    """Controls contain physical inputs and evaluation-only labels/weights.

    Control masses and agreement weights never enter the training call. Caller
    establishes population identity/disjointness and the base feature ordering.
    Diagnostic threshold failures are returned as failures, never repaired by
    changing predictions or selecting alternate models.
    """
    required = {'agreement_a', 'agreement_b', 'correlation', 'weights_a',
                'weights_b', 'correlation_mass', 'quality'}
    if set(controls) != required:
        raise ValueError('Exactly the required evaluation controls must be supplied')
    populations = [query_inputs, controls['agreement_a'], controls['agreement_b'], controls['correlation']]
    sizes = []
    for population in populations:
        prepared = prepare_features(*population, excluded_column)
        sizes.append(len(prepared['base']))
    # Run the evaluator's full boundary contract before costly training. These
    # placeholder scores are discarded and never reported as model evidence.
    evaluate(labels, np.zeros(len(labels)), controls['quality'],
             np.zeros(sizes[1]), np.zeros(sizes[2]), controls['weights_a'],
             controls['weights_b'], np.zeros(sizes[3]), controls['correlation_mass'])
    merged = tuple(np.concatenate([np.asarray(population[column]) for population in populations], axis=0)
                   for column in range(6))
    result = execute(training_inputs, labels, mass, merged, excluded_column)
    ends = np.cumsum(sizes)
    primary, a, b, correlation = np.split(result['query_scores'], ends[:-1])
    metrics = evaluate(labels, result['oof_scores'], controls['quality'], a, b,
                       controls['weights_a'], controls['weights_b'], correlation,
                       controls['correlation_mass'])
    return dict(query_scores=primary, evaluation=metrics, execution=result,
                population_sizes=tuple(sizes))
