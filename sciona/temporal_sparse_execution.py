"""Private, bounded source preparation and complete temporal sparse execution."""
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from sciona.temporal_sparse_features import TemporalSparseEncoder
from sciona.temporal_sparse_source import Source, events, inspect_source, snapshot, validate_time_splits
from sciona.temporal_sparse_training import fit, validate_controls


@dataclass(frozen=True, repr=False)
class Prepared:
    configuration: str = field(repr=False)
    sources: tuple[Source, ...] = field(repr=False)


def _configuration(payload):
    if type(payload) is not dict or set(payload) != {'version', 'sources', 'encoder_controls', 'controls', 'limits'}:
        raise ValueError('Invalid execution configuration')
    if type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('Unsupported configuration version')
    sources = payload['sources']
    if type(sources) is not dict or set(sources) != {'training', 'validation', 'query'}:
        raise ValueError('Invalid source roles')
    if any(type(p) is not str or not p for p in sources.values()):
        raise ValueError('Invalid source configuration')
    encoder = payload['encoder_controls']
    if type(encoder) is not dict or set(encoder) != {'hash_size', 'lookback', 'period', 'max_entities', 'max_history', 'max_pending'}:
        raise ValueError('Invalid encoder controls')
    limits = payload['limits']
    if type(limits) is not dict or set(limits) != {'max_line_bytes', 'max_source_bytes', 'max_output_rows'}:
        raise ValueError('Invalid execution limits')
    if any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError('Positive integer execution limits required')
    try:
        validate_controls(payload['controls'])
        TemporalSparseEncoder(**encoder)
        return json.dumps(payload, allow_nan=False, sort_keys=True)
    except (OverflowError, TypeError, ValueError):
        raise ValueError('Invalid execution controls') from None


def prepare(payload):
    configuration = _configuration(payload)
    config = json.loads(configuration)
    limits = config['limits']
    sources = tuple(inspect_source(config['sources'][role], labeled=role != 'query',
                    max_line_bytes=limits['max_line_bytes'], max_bytes=limits['max_source_bytes'])
                    for role in ('training', 'validation', 'query'))
    validate_time_splits(*sources, gap=config['controls']['gap'])
    if sources[2].rows > limits['max_output_rows']:
        raise ValueError('Prediction output exceeds configured limit')
    return Prepared(configuration, sources)


def execute(prepared):
    if type(prepared) is not Prepared:
        raise ValueError('Prepared execution required')
    try:
        config = json.loads(prepared.configuration)
        if _configuration(config) != prepared.configuration:
            raise ValueError('Invalid prepared configuration')
    except (TypeError, ValueError):
        raise ValueError('Invalid prepared configuration') from None
    if type(prepared.sources) is not tuple or len(prepared.sources) != 3 or any(type(s) is not Source for s in prepared.sources):
        raise ValueError('Invalid prepared sources')
    limits = config['limits']
    for role, source in zip(('training', 'validation', 'query'), prepared.sources):
        if (source.path != str(Path(config['sources'][role]).absolute()) or
            type(source.labeled) is not bool or source.labeled != (role != 'query') or
            type(source.max_line_bytes) is not int or source.max_line_bytes != limits['max_line_bytes'] or
            type(source.max_bytes) is not int or source.max_bytes != limits['max_source_bytes']):
            raise ValueError('Prepared source configuration changed')
    with ExitStack() as stack:
        # Verify every immutable byte snapshot before any estimator is fitted.
        streams = [stack.enter_context(snapshot(s)) for s in prepared.sources]
        verified = []
        for source, stream in zip(prepared.sources, streams):
            count = 0
            first = last = None
            for event in events(stream, labeled=source.labeled, max_line_bytes=source.max_line_bytes):
                if first is None:
                    first = event['time']
                last = event['time']
                count += 1
            if not count or (count, first, last) != (source.rows, source.first_time, source.last_time):
                raise ValueError('Prepared source summary changed')
            verified.append(source)
            stream.seek(0)
        validate_time_splits(*verified, gap=config['controls']['gap'])
        if verified[2].rows > limits['max_output_rows']:
            raise ValueError('Prediction output exceeds configured limit')
        inputs = [events(stream, labeled=s.labeled, max_line_bytes=s.max_line_bytes)
                  for s, stream in zip(verified, streams)]
        model = fit(inputs[0], encoder_controls=config['encoder_controls'], controls=config['controls'])
        validation = model.evaluate(inputs[1])
        predictions = []
        for values, _ in model.predict_batches(inputs[2]):
            if len(predictions) + len(values) > limits['max_output_rows']:
                raise ValueError('Prediction output exceeds configured limit')
            predictions.extend(values.tolist())
        if not all(math.isfinite(v) for v in predictions):
            raise ValueError('Nonfinite predictions')
        return dict(training_rows=model.training_rows, validation=validation, predictions=predictions,
                    query_rows=len(predictions), updates_per_model=model.updates_per_model,
                    models=len(model.models), feature_count=8 + config['encoder_controls']['hash_size'])
