"""Synthetic feature/reducer probes of the pinned OpenVaccine inference source."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    cache=Path('/private/tmp/sciona_openvaccine_source');manifest=json.loads((cache/'manifest.json').read_text())
    name='scripts/nullrecurrent_inference.py';raw=(cache/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==next(p['sha256'] for p in manifest['pins'] if p['software_path']==name)
    wanted={'calc_neighbor','get_distance_matrix_2d','get_distance_matrix','reverse_input','reverse_BBP_3D','get_preds_df'}
    definitions=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert len(definitions)==len(wanted)
    ns=dict(np=np,pd=pd);exec(compile(ast.Module(body=definitions,type_ignores=[]),'<pinned-inference-primitives>','exec'),ns)
    adjacency=np.zeros((1,8,8,1))
    for i,j in [(0,7),(2,5)]:adjacency[0,i,j,0]=adjacency[0,j,i,0]=1
    a,b=np.indices((8,8));distance=np.full((8,8),10.)
    for k in range(8):distance=np.minimum(distance,abs(a-k)+abs(b-k))
    for i,j in zip(*np.where(adjacency[0,:,:,0]==1)):
        distance=np.minimum(distance,1+abs(a-i)+abs(b-j))
    expected=np.stack([(1/(1+distance))**power for power in (1,2,4)],axis=-1)[None]
    np.testing.assert_array_equal(ns['get_distance_matrix_2d'](adjacency),expected)
    linear=np.stack([(1/(1+abs(a-b)))**power for power in (1,2,4)],axis=-1)[None]
    np.testing.assert_array_equal(ns['get_distance_matrix'](adjacency),linear)
    nodes=np.arange(24).reshape(1,8,3)
    np.testing.assert_array_equal(ns['reverse_input'](ns['reverse_input'](nodes)),nodes)
    np.testing.assert_array_equal(ns['reverse_BBP_3D'](ns['reverse_BBP_3D'](adjacency)),adjacency)
    first=pd.DataFrame(np.full((2,5),10.));second=pd.DataFrame(np.zeros((2,5)))
    output=ns['get_preds_df']([first,second])
    reverse=ns['get_preds_df']([pd.DataFrame(np.zeros((2,5))),pd.DataFrame(np.full((2,5),10.))])
    assert np.all(output.iloc[:,1:].values==5) and np.all(reverse.iloc[:,1:].values==3)
    base=pd.DataFrame(np.ones((2,5)));other=pd.DataFrame(np.full((2,5),10.))
    ns['get_preds_df']([base,other])
    assert np.all(base.values==7) and np.all(other.values==6)
    tree=ast.parse(raw)
    fit_calls=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','fit_generator','train_on_batch') for n in ast.walk(tree))
    assert fit_calls==0
    entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_preds')
    ns['_make_pred']=lambda sequence,output_feature:None
    exec(compile(ast.Module(body=[entry],type_ignores=[]),'<source-entrypoint>','exec'),ns)
    try:ns['make_preds'](['synthetic-placeholder'],'synthetic-output')
    except TypeError as error:assert 'output_feature' in str(error)
    else:raise AssertionError('Missing output selector forwarding not reproduced')
    report=dict(training_fit_calls=fit_calls,make_preds_missing_output_selector_reproduced=True,status='passed' ,source_commit=manifest['commit'],distance_features_exact=True,
        distance_semantics='Capped Manhattan propagation from diagonal-zero and pair-one seeds; not general graph shortest paths.',
        double_reversal_exact=True,first_ensemble_member_unclipped=True,ensemble_order_dependence_reproduced=True,
        reducer_mutates_input_frames=True,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Synthetic feature and reducer primitives only; no full Keras model, folding, pretraining or supervised/pseudo-label lifecycle claim.')
    (ROOT/'docs/reviews/competition_openvaccine_source_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
