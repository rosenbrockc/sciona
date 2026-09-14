"""Strict named Slim tensor -> Torch state conversion, without checkpoint I/O.

Callers supply decoded CPU tensors under source variable names (without :0).
No automatic download, pickle loading or claim of checkpoint binary decoding.
"""
import torch

SCOPES = {
    'inception_v3': ('InceptionV3','AdvInceptionV3','Ens3AdvInceptionV3','Ens4AdvInceptionV3'),
    'inception_v4': ('InceptionV4',),
    'inception_resnet': ('InceptionResnetV2','EnsAdvInceptionResnetV2'),
    'resnet101': ('resnet_v2_101',),
}


def mapping(model, family, scope):
    if family not in SCOPES or scope not in SCOPES[family]:
        raise ValueError('source family/scope combination required')
    result = {}
    def add(local, source, layout='identity'):
        if local in result or any(v[0]==source for v in result.values()):
            raise ValueError('duplicate state mapping')
        result[local] = (scope+'/'+source, layout)
    def conv(local, source, biased):
        add(local+'.weight',source+'/weights','HWIO')
        if biased: add(local+'.bias',source+'/biases')
    def norm(local, source, scale):
        for name in ('beta','moving_mean','moving_variance') + (('gamma',) if scale else ()):
            add(local+'.'+name, source+'/'+name)
    if family!='resnet101':
        expected_root = SCOPES[family][0]
        for i,node in enumerate(model.topology['nodes']):
            if not node['name'].startswith(expected_root+'/'): raise ValueError('model topology family mismatch')
            source = node['name'][len(expected_root)+1:]
            local = 'layers.'+str(i)
            if node['op']=='conv':
                conv(local+'.conv',source,not node['normalized'])
                if node['normalized']: norm(local+'.norm',source+'/BatchNorm',False)
            elif node['op']=='linear':
                add(local+'.weight',source+'/weights','IO')
                add(local+'.bias',source+'/biases')
    else:
        conv('root','conv1',True)
        for b,block in enumerate(model.blocks):
            for u,unit in enumerate(block):
                local=f'blocks.{b}.{u}'
                source=f'block{b+1}/unit_{u+1}/bottleneck_v2'
                norm(local+'.preact',source+'/preact',True)
                if unit.shortcut is not None:conv(local+'.shortcut',source+'/shortcut',True)
                for k in (1,2,3):conv(local+f'.conv{k}',source+f'/conv{k}',k==3)
                for k in (1,2):norm(local+f'.bn{k}',source+f'/conv{k}/BatchNorm',True)
        norm('postnorm','postnorm',True)
        conv('logits','logits',True)
    if result.keys()!=model.state_dict().keys():raise ValueError('mapping does not cover exact model state')
    if len({v[0] for v in result.values()})!=len(result):raise ValueError('nonunique source variable names')
    return result


def convert_state(model, tensors, *, family, scope):
    """Return owned contiguous tensors; reject extras, missing keys and bad state."""
    names=mapping(model,family,scope)
    if not isinstance(tensors,dict) or tensors.keys()!={v[0] for v in names.values()}:
        raise ValueError('exact complete source model variable set required')
    expected=model.state_dict();result={}
    for key,(source,layout) in names.items():
        value=tensors[source]
        if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.layout!=torch.strided or value.dtype!=expected[key].dtype
                or not torch.isfinite(value).all()):
            raise ValueError('finite CPU tensor of exact model dtype required')
        if layout=='HWIO':
            if value.ndim!=4:raise ValueError('HWIO convolution tensor required')
            value=value.permute(3,2,0,1)
        elif layout=='IO':
            if value.ndim!=2:raise ValueError('IO dense tensor required')
            value=value.T
        if value.shape!=expected[key].shape:raise ValueError('source tensor shape mismatch')
        if key.endswith('moving_variance') and torch.any(value<0):
            raise ValueError('nonnegative normalization variance required')
        result[key]=value.detach().clone(memory_format=torch.contiguous_format)
    return result
