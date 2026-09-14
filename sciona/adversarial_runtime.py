"""Versioned private JSON boundary for complete RGB attacks.

Numerical transport reuses the immutable validated codec; no file paths, pickle,
model downloads or payload logging. Decoded states are validated against the
exact model at execution. Payloads and outputs must remain private runtime data.
"""
import base64
import math
import numpy as np
from sciona.dfdc_codec import decode_array,decode_state,encode_array
from sciona.adversarial_ensemble import ORDER
from sciona.adversarial_updates import configuration
from sciona.adversarial_rgb_attack import attack_rgb

IMAGE_BYTES=64*1024**2
STATE_BYTES=2*1024**3


def keys(value, required):
    if not isinstance(value,dict) or value.keys()!=set(required):
        raise ValueError('exact adversarial runtime fields required')


def prepare(payload):
    keys(payload,('version','rgb','targets','config','initializations'))
    if type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('unsupported runtime version')
    config=payload['config']
    keys(config,('mode','epsilon','momentum','non_targeted_iterations'))
    selected=configuration(config['mode'],config['epsilon'],non_targeted_iterations=config['non_targeted_iterations'])
    if (type(config['non_targeted_iterations']) is not int or config['non_targeted_iterations']<=0
            or config['mode']=='targeted' and config['non_targeted_iterations']!=10):
        raise ValueError('targeted branch must retain unused non-targeted default')
    if type(config['momentum']) not in (int,float) or not math.isfinite(config['momentum']) or config['momentum']<0:
        raise ValueError('finite nonnegative momentum required')
    rgb=decode_array(payload['rgb'],max_bytes=IMAGE_BYTES)
    if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[1:]!=(299,299,3) or len(rgb)<1:
        raise ValueError('nonempty299RGBuint8 payload required')
    targets=None
    if config['mode']=='targeted':
        targets=decode_array(payload['targets'],max_bytes=IMAGE_BYTES)
        if targets.dtype!=np.int64 or targets.shape!=(len(rgb),) or np.any(targets<0) or np.any(targets>=1001):
            raise ValueError('valid per-image int64 target classes required')
    elif payload['targets'] is not None:raise ValueError('non-targeted targets must be null')
    specs=payload['initializations'];keys(specs,ORDER[selected['branch']])
    initializations={};remaining=STATE_BYTES
    for scope,spec in specs.items():
        if not isinstance(spec,dict) or spec.get('kind') not in ('random','slim_tensors'):
            raise ValueError('explicit model initialization required')
        keys(spec,('kind','seed') if spec['kind']=='random' else ('kind','seed','tensors'))
        if type(spec['seed']) is not int or not 0<=spec['seed']<2**32:
            raise ValueError('portable uint32 initialization seed required')
        decoded={'kind':spec['kind'],'seed':spec['seed']}
        if spec['kind']=='slim_tensors':
            tensors=decode_state(spec['tensors'],max_bytes=max(1,remaining))
            size=sum(v.numel()*v.element_size() for v in tensors.values())
            if size>remaining:raise ValueError('aggregate ensemble state limit exceeded')
            if any(v.numpy().dtype!=np.float32 for v in tensors.values()):raise ValueError('Slim model variables must be float32')
            remaining-=size;decoded['tensors']=tensors
        initializations[scope]=decoded
    return {'rgb':rgb,'targets':targets,'config':dict(config),'initializations':initializations}


def execute(prepared):
    keys(prepared,('rgb','targets','config','initializations'))
    result=attack_rgb(prepared['rgb'],targets=prepared['targets'],
                      initializations=prepared['initializations'],**prepared['config'])
    return {'version':1,'normalized_images':encode_array(result['images']),
            'labels':encode_array(result['labels']),
            'pngs':[base64.b64encode(data).decode('ascii') for data in result['pngs']],
            'batch_losses':result['batch_losses'],
            'saved_pixel_bound_guaranteed':False,
            'initialization_kinds':{s:v['kind'] for s,v in prepared['initializations'].items()}}


def run(payload):
    return execute(prepare(payload))
