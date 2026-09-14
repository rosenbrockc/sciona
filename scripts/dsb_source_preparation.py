"""Pinned raw preparation oracle with synthetic array IO replacing file access."""
import ast
from types import SimpleNamespace
import numpy as np
import scipy.ndimage
from scipy.ndimage import zoom,binary_dilation,generate_binary_structure
from skimage import measure
from skimage.morphology import convex_hull_image
from scripts.audit_competition_dsb_semantics import checked_source


class ArrayPreparation(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':
            return ast.copy_location(ast.Pass(),node)
        return self.generic_visit(node)
    def visit_Assign(self,node):
        expression=ast.unparse(node)
        if expression=='resolution = np.array([1, 1, 1])':
            return ast.copy_location(ast.parse('resolution = requested_resolution.copy()').body[0],node)
        if expression=='extramask = dilatedMask - Mask':
            return ast.copy_location(ast.parse('extramask = dilatedMask.astype(int) - Mask.astype(int)').body[0],node)
        return self.generic_visit(node)


class SourcePreparation:
    def __init__(self,source_root,pins):
        tree=ast.parse(checked_source(source_root,pins,'preprocessing/step1.py'))
        names={'binarize_per_slice','all_slice_analysis','fill_hole','two_lung_only','step1_python'}
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        halves=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute) and node.attr=='in1d':node.attr='isin'
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.right,ast.Constant) and node.right.value==2:
                node.op=ast.FloorDiv();halves+=1
        assert halves==8
        segmentation=dict(np=np,scipy=scipy,measure=measure,load_scan=lambda value:value,get_pixels_hu=lambda value:value)
        exec(compile(tree,'<source-array-segmentation>','exec'),segmentation)
        self.segment=segmentation['step1_python']
        tree=ast.parse(checked_source(source_root,pins,'training/prepare.py'))
        names={'process_mask','lumTrans','resample','worldToVoxelCoord','savenpy','savenpy_luna'}
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.code=compile(ast.fix_missing_locations(ArrayPreparation().visit(tree)),'<source-array-preparation>','exec')

    def __call__(self,record):
        data=record['preparation'];saved=[];loads=[]
        image=np.asarray(data['volume']);spacing=np.asarray(data['spacing'],dtype=float)
        rows=np.asarray(data['annotations_xyz_diameter'],dtype=float)
        proxy=SimpleNamespace(**{key:getattr(np,key) for key in dir(np) if not key.startswith('_')})
        proxy.save=lambda _,value:saved.append(value.copy())
        scope=dict(np=proxy,zoom=zoom,binary_dilation=binary_dilation,
            generate_binary_structure=generate_binary_structure,convex_hull_image=convex_hull_image,
            requested_resolution=np.asarray(data.get('requested_spacing',(1.,1.,1.))),
            os=SimpleNamespace(path=SimpleNamespace(join=lambda *args:'synthetic')))
        if record['kind']=='world':
            left=data['left_mask'];right=data['right_mask']
            if np.any(left & right):raise ValueError('source labeled mask requires disjoint components')
            segmentation=left.astype(np.int16)*3+right.astype(np.int16)*4
            def load(_):
                loads.append(True)
                return ((segmentation if len(loads)==1 else image).copy(),np.asarray(data['origin_zyx']),spacing,data.get('flip',False))
            scope['load_itk_image']=load
            annos=np.column_stack((np.zeros(len(rows)),rows))
        elif record['kind']=='voxel':
            scope['step1_python']=lambda _:self.segment((image.copy(),spacing))
            annos=np.empty((len(rows),5),dtype=object);annos[:,0]='0';annos[:,1:]=rows
        else:raise ValueError('unknown source preparation branch')
        exec(self.code,scope)
        if record['kind']=='world':scope['savenpy_luna'](0,annos,['0'],'synthetic','synthetic','synthetic')
        else:scope['savenpy'](0,annos,['0'],'synthetic','synthetic')
        assert len(saved)==2
        return saved[0],np.empty((0,4),dtype=float) if np.all(saved[1]==0) else saved[1]
