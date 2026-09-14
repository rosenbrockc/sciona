"""Audit pinned pseudo preparation variants at the decoded-array boundary."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def dump(nodes):
    return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)


def method(tree,cls,name):
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls]
    assert len(classes)==1
    nodes=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1
    return nodes[0]


def suffix_after_raster(node):
    starts=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and ast.unparse(n).startswith('self.data = rasterio.open(')]
    assert len(starts)==1
    return node.body[starts[0]+1:]


def validate(root,source):
    all_pins={}
    for kind in ['source','loader','pseudo']:
        all_pins.update(json.loads((root/f'docs/reviews/competition_hubmap_{kind}_pins.json').read_text())['files'])
    for name,digest in all_pins.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    raw=ast.parse((source/'src/utils_data_generation.py').read_text())
    for name in ['__len__','__getitem__']:
        assert dump([method(raw,'HuBMAPDataset',name)])==dump([method(raw,'HuBMAPDatasetExternal',name)])
    assert dump(suffix_after_raster(method(raw,'HuBMAPDataset','__init__')))==dump(suffix_after_raster(method(raw,'HuBMAPDatasetExternal','__init__')))
    base=source/'src/03_generate_pseudo_labels'
    folders=sorted(base.iterdir())
    primary=ast.parse((folders[0]/'dataset.py').read_text())
    loop=ast.parse((folders[0]/'utils_inference.py').read_text())
    original_loop=next(n for n in loop.body if isinstance(n,ast.FunctionDef) and n.name=='get_pred_mask')
    for folder in folders[1:]:
        tree=ast.parse((folder/'dataset.py').read_text())
        for name in ['__len__','__getitem__']:
            assert dump([method(primary,'HuBMAPDataset',name)])==dump([method(tree,'HuBMAPDatasetExternal',name)])
        assert dump(suffix_after_raster(method(primary,'HuBMAPDataset','__init__')))==dump(suffix_after_raster(method(tree,'HuBMAPDatasetExternal','__init__')))
        tree=ast.parse((folder/'utils_inference_external.py').read_text())
        external=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_pred_mask_external')
        # Only the first Dataset construction statement differs; all subsequent
        # batching, model calls, averaging and stitching statements must match.
        assert dump(original_loop.body[1:])==dump(external.body[1:])
        def recipe(path):
            return dump([n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.Assign))])
        assert recipe(folders[0]/'transforms.py')==recipe(folder/'transforms.py')
    initial=source/'src/01_data_preparation/01_01/data_preparation_01_01.py'
    def sorting(path):
        nodes=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Assign)
               and any(ast.unparse(n).startswith(s) for s in ['bs, sz, sz, c =','idxs = np.argsort(','img_patches = img_patches[idxs]','mask_patches = mask_patches[idxs]'])]
        assert len(nodes)==4
        return dump(sorted(nodes,key=lambda n:n.lineno))
    stages=[]
    for folder in sorted((source/'src/04_data_preparation_pseudo_label').iterdir()):
        path=next(folder.glob('*.py'));tree=ast.parse(path.read_text())
        assert sorting(path)==sorting(initial)
        config={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                for key,value in zip(node.keys,node.values):
                    if isinstance(key,ast.Constant) and key.value in {'tile_size','shift_h','shift_w'}:
                        config[key.value]=ast.literal_eval(value)
        number=int(folder.name.split('_')[1])
        assert config==dict(tile_size=1024,shift_h=512 if number%2==0 else 0,shift_w=512 if number%2==0 else 0)
        writers=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Call)
                 and isinstance(n.func.func,ast.Name) and n.func.func.id=='delayed']
        assert len(writers)==1 and ast.unparse(writers[0])=='delayed(generate_data)(filename, i, x, y, config)'
        stages.append(folder.name[:5])
    nodes=[n for n in ast.parse((source/'src/utils.py').read_text()).body
           if isinstance(n,ast.FunctionDef) and n.name in {'mask2rle','rle2mask'}]
    assert len(nodes)==2
    ns=dict(np=np,cv2=cv2)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-pseudo-mask-handoff>','exec'),ns)
    cases=0
    for h,w in [(1,1),(7,11),(32,32),(513,519)]:
        for kind in ['empty','full','sparse','checker']:
            mask=np.zeros((h,w),dtype=np.uint8)
            if kind=='full':mask.fill(1)
            if kind=='sparse':mask[-1,-1]=1
            if kind=='checker':mask=(np.indices((h,w)).sum(0)%2).astype(np.uint8)
            encoded=ns['mask2rle'](mask,(h,w),0)
            np.testing.assert_array_equal(ns['rle2mask'](encoded,(h,w)),mask)
            cases+=1
    paths=['scripts/validate_hubmap_pseudo_orchestration.py']+[
        f'docs/reviews/competition_hubmap_{kind}_pins.json' for kind in ['source','loader','pseudo']]
    return dict(approved=False,synthetic_only=True,preparation_stages=stages,
                external_raw_tiler_numerical_ast_identical=True,external_inference_variants_numerical_ast_identical=2,
                exact_binary_mask_rle_handoff_cases=cases,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Numerical source equivalence after raster-open boundary; external file lookup, codecs and subdataset IO excluded.',
                             'Eight preparation stages audited for grid configs, sort and shared writer; not full script execution.',
                             'Binary masks with source small-mask threshold zero; no resized mask handoff claimed.',
                             'Complete inference-to-retraining lifecycle and final graph publication remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_pseudo_orchestration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
