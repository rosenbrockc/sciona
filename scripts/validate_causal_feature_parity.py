#!/usr/bin/env python3
"""Compare reconstructed features with a pinned, mechanically ported Python 2 source."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import operator
import importlib.metadata
import inspect
from pathlib import Path
import types
import warnings
import numpy as np
from sciona.atoms.causal_inference.feature_primitives.selected_features import FEATURE_ORDER,extract_causal_selected_features

HASHES={'features.py':'1f70b47017370c2642c31cfe3a3727d5cd29d7743f8de538a186afdf8d1b4e5e','estimator.py':'a685f8c7afad368afe78191fd8647381c8d262201d7b4411fe9032c205b2d070','hsic.py':'e11fc510d45ebdd3ef1f67ba99604e45628e28d4abd618bd0c0ab50dbbd337b6'}
COMMIT='f4d0f0d8cd160ce0d39a88c0bdc81c1723024acd'


class Python2Collections(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='iterkeys':node.func.attr='keys'
        if (isinstance(node.func,ast.Attribute) and node.func.attr in {'keys','values','items'}) or (isinstance(node.func,ast.Name) and node.func.id in {'zip','map','range'}):
            return ast.copy_location(ast.Call(func=ast.Name(id='list',ctx=ast.Load()),args=[node],keywords=[]),node)
        return node


def load_reference(directory):
    for name,digest in HASHES.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('pinned upstream hash mismatch')
    hsic=types.ModuleType('reference_hsic')
    exec(compile((directory/'hsic.py').read_text(),'pinned_hsic.py','exec'),hsic.__dict__)
    tree=ast.parse((directory/'features.py').read_text())
    tree.body=[node for node in tree.body if not (isinstance(node,ast.Import) and any(a.name in {'hsic','operator'} for a in node.names))]
    tree=ast.fix_missing_locations(Python2Collections().visit(tree))
    ns={'hsic':hsic,'operator':types.SimpleNamespace(**vars(operator),div=operator.truediv)}
    exec(compile(tree,'ported_pinned_features.py','exec'),ns)
    selected=next(ast.literal_eval(node.value) for node in ast.parse((directory/'estimator.py').read_text()).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='selected_features' for t in node.targets))
    definitions={}
    for name,inputs,transform in ns['all_features']:
        key=name.lstrip('+')+'['+(','.join(inputs) if isinstance(inputs,list) else inputs)+']'
        definitions[key]=(inputs,transform.transformer)
    if len(selected)!=len(FEATURE_ORDER):raise ValueError('selected feature count differs')
    def extract(x,tx,y,ty):
        values={'A':x.copy(),'B':y.copy(),'A type':tx,'B type':ty}
        def compute(key):
            if key not in values:
                inputs,fn=definitions[key]
                values[key]=fn(*(compute(k) for k in inputs)) if isinstance(inputs,list) else fn(compute(inputs))
            return values[key]
        return np.asarray([compute(name) for name in selected],dtype=float)
    return extract


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference-dir',required=True,type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    reference=load_reference(args.reference_dir)
    counts=Counter();mismatches=Counter();maximum={}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for seed in range(3):
            for n in [64,127,256]:
                for tx in ['Numerical','Categorical','Binary']:
                    for ty in ['Numerical','Categorical','Binary']:
                        rng=np.random.default_rng(3000+seed*1000+n)
                        x=rng.normal(size=n) if tx=='Numerical' else rng.integers(0,2 if tx=='Binary' else 4,size=n).astype(float)
                        y=.3*x+rng.normal(size=n) if ty=='Numerical' else rng.integers(0,2 if ty=='Binary' else 3,size=n).astype(float)
                        for a,ta,b,tb in [(x,tx,y,ty),(y,ty,x,tx)]:
                            counts['orientations']+=1
                            try:expected=reference(a,ta,b,tb)
                            except Exception as exc:
                                counts['reference_error_'+type(exc).__name__]+=1;continue
                            if not np.all(np.isfinite(expected)):
                                counts['reference_nonfinite']+=1;continue
                            try:actual=extract_causal_selected_features([(a,b)],[(ta,tb)])[0]
                            except Exception as exc:
                                counts['implementation_error_'+type(exc).__name__]+=1;continue
                            counts['compared']+=1
                            close=np.isclose(actual,expected,rtol=1e-9,atol=1e-10)
                            counts['all_features_match']+=bool(close.all())
                            for i in np.flatnonzero(~close):
                                name=FEATURE_ORDER[i];mismatches[name]+=1;maximum[name]=max(maximum.get(name,0),float(abs(actual[i]-expected[i])))
    from sciona.atoms.causal_inference.feature_primitives import atoms,selected_features
    from sciona.atoms.causal_inference.conditional_statistics import atoms as conditional,source_entropy
    implementation_hashes={m.__name__:hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest() for m in [atoms,selected_features,conditional,source_entropy]}
    report={'synthetic_only':True,'implementation_hashes':implementation_hashes,'library_versions':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn']},'source_commit':COMMIT,'source_hashes':HASHES,'port':'Python 2 collection materialization, operator.div -> true division; current installed scientific libraries retained','counts':dict(counts),'mismatch_counts':dict(mismatches),'maximum_absolute_error_by_feature':maximum}
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
