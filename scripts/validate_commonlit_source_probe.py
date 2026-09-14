"""Reproduce CommonLit source inconsistencies using synthetic arrays only."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    pins=json.loads((args.source/'code_pins.json').read_text())
    for pin in pins['pins']:
        raw=(args.source/pin.get('extracted_path',pin['software_path'])).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pin['sha256']
    def cells(name):
        return {c['cell_index']:c['source'] for c in json.loads((args.source/'notebooks'/f'{name}.code.json').read_text())}
    inference=cells('05_clrp_inference')
    labeling=cells('03_clrp_external_data_labeling')
    ns={'np':np}
    for text in (labeling[12],inference[14]):
        nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
        assert len(nodes)==1
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-functions>','exec'),ns)
    selected,predictions=ns['filter_on_stdev'](['synthetic-a','synthetic-b','synthetic-c'],[.5,1.,2.],[0.,0.,2.],[1.,1.,0.])
    assert selected==['synthetic-a'] and predictions==[.5]
    # Original inference ensemble2 passes six columns into a five-feature
    # ridge head according to the included training notebook invocation.
    rng=np.random.default_rng(1729)
    ridge5=Ridge(alpha=1.).fit(rng.normal(size=(12,5)),rng.normal(size=12))
    ns['load']=lambda _:ridge5
    try:
        ns['make_ensembler_predictions']([[np.zeros(3) for _ in range(6)]],['synthetic-head'])
    except ValueError as error:
        assert 'expecting 5 features' in str(error)
    else: raise AssertionError('Source five/six feature mismatch not reproduced')
    constants={name:name for name in ('ALBERT_TRAINED_1','DEBERTA_TRAINED_1','ALBERT_TRAINED_2','DEBERTA_TRAINED_2','ROBERTA_TRAINED_1','ELECTRA_TRAINED_1','ALBERT_TRAINED_3','DEBERTA_TRAINED_3')}
    constants.update(os=os,np=np,test_tx=['synthetic'],predict_fast=lambda **_: [0.])
    try: exec(compile(inference[18],'<source-inference-call-site>','exec'),constants)
    except AttributeError as error:
        assert 'joun' in str(error)
    else: raise AssertionError('Original path typo not reproduced')
    arrays={'np':np,'ensemble_1_preds':np.array([1.,2.]),'ensemble_2_preds':np.array([3.,4.]),
        'albert_single_preds':np.array([5.,6.]),'deberta_bs_0':np.array([7.,8.]),'deberta_bs_1':np.array([9.,10.])}
    exec(compile(inference[21]+'\n'+inference[22],'<source-final-blend>','exec'),arrays)
    expected=3/8*np.array([1.,2.])+2/8*np.array([3.,4.])+3/8*(.65*np.array([5.,6.])+.175*np.array([7.,8.])+.175*np.array([9.,10.]))
    np.testing.assert_allclose(arrays['final_predictions'],expected,rtol=0,atol=1e-15)
    args.output.write_text(json.dumps({'source_commit':pins['commit'],'checks':{
        'strict_standard_error_filter':True,'five_vs_six_ridge_features_reproduced':True,
        'source_path_typo_reproduced':True,'final_blend_matches_scalar_weights':True},
        'scope':'Selected original source functions/call sites with synthetic arrays and stub prediction I/O; no full training, neural topology, or promotion claim.',
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Four source checks passed; two execution defects reproduced')


if __name__=='__main__':main()
