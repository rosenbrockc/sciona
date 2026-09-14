"""Write only generated synthetic streams into a caller-owned temporary directory."""
import json

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

