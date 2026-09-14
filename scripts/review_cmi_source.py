"""Review pinned public CMI software without executing notebook cells or reading inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row


def review(root: Path, source_root: Path) -> dict:
    pin = json.loads((root / 'docs/reviews/competition_cmi_notebook_pin.json').read_text())
    raw = (source_root / 'code_cells.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['code_cells_sha256']:
        raise ValueError('Cached public source differs from recorded pin')
    cells = json.loads(raw)
    inventory = []
    parsed = {}
    for cell in cells:
        # IPython directives are not Python statements. Shell continuation cells
        # remain unparsed and are explicitly recorded; nothing is executed.
        source = '\n'.join(line for line in cell['source'].splitlines()
                           if not line.lstrip().startswith(('!', '%')))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            inventory.append({'cell': cell['index'], 'parsed': False})
            continue
        parsed[cell['index']] = tree
        inventory.append({'cell': cell['index'], 'parsed': True,
                          'definitions': [n.name for n in ast.walk(tree)
                                          if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                          'imports': sorted({n.module for n in ast.walk(tree)
                                             if isinstance(n, ast.ImportFrom) and n.module})})
    config = {n.targets[0].id: ast.literal_eval(n.value)
              for n in parsed[53].body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)}
    weights = list(config['shimacos_model_name2weight'].values()) + list(config['sakami_model_name2weight'].values())
    assert len(weights) == 5 and abs(sum(weights) - 1) < 1e-12
    version = 'a4400b90-e996-55a4-ab83-c99de669c5fe'
    expected = 'c02e81e4444ec844fce7efcdd609197adcebf9b5aa66b01a5cd46c76993b0a80'
    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
                         row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        row = db.execute('SELECT a.status,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (version,)).fetchone()
        assert row['content_hash'] == expected and row['status'] == 'draft'
    return {
        'read_only': True, 'approved': False, 'source_version_id': version,
        'source_hash': expected, 'notebook_version': pin['version'],
        'code_cells_sha256': pin['code_cells_sha256'], 'inventory': inventory,
        'findings': [
            'Intake has unrolling, rolling statistics, U-Net, transformer smoothing and peak finding; this does not describe the full multi-branch submission.',
            'Active ensemble configuration has three shimacos stacking branches (LightGBM, CatBoost, neural) and two sakami branches (transformer, CNN). Five weights sum to one.',
            'Cell 55 is commented-out alternative code. Its 1:3:2 weights are not the active final ensemble configuration.',
            'Cell 38 defines event postprocessing and chunk creation; cells 39-44 define feature generation; cell 49 includes residual GRU and CNN/RNN stacking classes.',
            'Notebook imports external settings, utils and dynamically loaded model code; complete source closure and training lifecycle are not yet established.',
            'Specific notebook page displays Apache 2.0. Indexed page displays version 1 while retrieved source is version 3; inherited software attribution remains to review.'
        ],
        'checks': {'cached_code_hash': True, 'original_intake_hash_and_draft': True,
                   'active_ensemble_branches': len(weights), 'active_weights_sum': sum(weights)},
        'next_action': 'Resolve external public software dependencies and training source; validate the actual computational path on synthetic inputs before Tier 3 publication.',
        'confidentiality': 'Only public software inspected. No observations, weights, input-source metadata or notebook outputs retained in this report.'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = review(root, args.source_root)
    (root / 'docs/reviews/competition_cmi_source_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['checks']))
