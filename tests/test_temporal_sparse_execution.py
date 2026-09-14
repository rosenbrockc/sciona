from dataclasses import replace
import json
import pytest
from sciona import temporal_sparse_execution as execution


def payload(tmp_path):
    sources = {}
    for role, start, stop in [('training', 0, 30), ('validation', 30, 40), ('query', 40, 42)]:
        path = tmp_path / (role + '.jsonl')
        rows = []
        for i in range(start, stop):
            row = dict(entity=f'e{i%2}', time=float(i), value=float(i%2), category=f'c{i%2}')
            if role != 'query':
                row['target'] = 1. + i % 2
            rows.append(json.dumps(row))
        path.write_text('\n'.join(rows) + '\n')
        sources[role] = str(path)
    return dict(version=1, sources=sources,
                encoder_controls=dict(hash_size=16, lookback=8., period=12., max_entities=4, max_history=10, max_pending=4),
                controls=dict(chunk_size=7, seed=12, learning_rate=.005, alpha=.001, feature_clip=3., max_prediction=20., gap=0.),
                limits=dict(max_line_bytes=4096, max_source_bytes=65536, max_output_rows=2))


def forbidden(*args, **kwargs):
    raise AssertionError('Fitting must not start')


def test_complete_execution_and_repeat(tmp_path):
    prepared = execution.prepare(payload(tmp_path))
    result = execution.execute(prepared)
    assert result == execution.execute(prepared)
    assert result['training_rows'] == 30 and result['validation']['rows'] == 10
    assert result['query_rows'] == 2 and result['models'] == 2
    assert result['updates_per_model'] == 5 and result['feature_count'] == 24
    assert all(0 <= v <= 20 for v in result['predictions'])
    assert str(tmp_path) not in repr(prepared) + json.dumps(result)


@pytest.mark.parametrize('role', ['training', 'validation', 'query'])
def test_changed_source_rejected_before_fit(tmp_path, monkeypatch, role):
    config = payload(tmp_path)
    prepared = execution.prepare(config)
    with open(config['sources'][role], 'a') as stream:
        stream.write('\n')
    monkeypatch.setattr(execution, 'fit', forbidden)
    with pytest.raises(ValueError, match='changed'):
        execution.execute(prepared)


def test_output_limit_before_fit(tmp_path, monkeypatch):
    config = payload(tmp_path)
    config['limits']['max_output_rows'] = 1
    monkeypatch.setattr(execution, 'fit', forbidden)
    with pytest.raises(ValueError, match='output'):
        execution.prepare(config)


def test_forged_summary_rejected_before_fit(tmp_path, monkeypatch):
    prepared = execution.prepare(payload(tmp_path))
    altered = replace(prepared, sources=prepared.sources[:2] + (replace(prepared.sources[2], rows=1),))
    monkeypatch.setattr(execution, 'fit', forbidden)
    with pytest.raises(ValueError, match='summary'):
        execution.execute(altered)


def test_validation_labels_isolated(tmp_path):
    config = payload(tmp_path)
    first = execution.execute(execution.prepare(config))
    path = config['sources']['validation']
    with open(path) as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        row['target'] = 999.
    with open(path, 'w') as stream:
        stream.write('\n'.join(map(json.dumps, rows)) + '\n')
    second = execution.execute(execution.prepare(config))
    assert first['predictions'] == second['predictions']
    assert first['validation']['rmsle'] != second['validation']['rmsle']


def test_snapshots_closed_on_fit_failure(tmp_path, monkeypatch):
    from contextlib import contextmanager
    original = execution.snapshot
    streams = []
    @contextmanager
    def tracked(source):
        with original(source) as stream:
            streams.append(stream)
            yield stream
    monkeypatch.setattr(execution, 'snapshot', tracked)
    monkeypatch.setattr(execution, 'fit', forbidden)
    with pytest.raises(AssertionError):
        execution.execute(execution.prepare(payload(tmp_path)))
    assert len(streams) == 3 and all(s.closed for s in streams)


@pytest.mark.parametrize('change', ['extra', 'boolean', 'infinite', 'oversized'])
def test_invalid_configuration(tmp_path, change):
    config = payload(tmp_path)
    if change == 'extra': config['extra'] = True
    if change == 'boolean': config['limits']['max_output_rows'] = True
    if change == 'infinite': config['controls']['gap'] = float('inf')
    if change == 'oversized': config['encoder_controls']['period'] = 10**1000
    with pytest.raises(ValueError):
        execution.prepare(config)


def test_forged_size_limit_rejected_before_open(tmp_path, monkeypatch):
    prepared = execution.prepare(payload(tmp_path))
    altered = replace(prepared, sources=(replace(prepared.sources[0], max_bytes=-1),) + prepared.sources[1:])
    monkeypatch.setattr(execution, 'snapshot', forbidden)
    with pytest.raises(ValueError, match='configuration'):
        execution.execute(altered)
