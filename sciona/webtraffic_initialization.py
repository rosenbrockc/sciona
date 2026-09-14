"""Source s32 initializer transforms over explicit unit-uniform draws.

Draw generation is external: no TensorFlow op-seed/random-sequence claim.
"""
import math
import torch


def uniform_shapes(encoder_features,decoder_features):
    if type(encoder_features) is not int or type(decoder_features) is not int or min(encoder_features,decoder_features)<1:
        raise ValueError('Positive runtime feature depths required')
    depth=267
    return {**{f'encoder_{i}':(depth,encoder_features if i<3 else depth) for i in range(6)},
            'decoder_gate_kernel':(decoder_features+1+depth,2*depth),
            'decoder_candidate_kernel':(decoder_features+1+depth,depth),'projection_kernel':(depth,1)}


def initialize_from_uniforms(draws,encoder_features,decoder_features):
    shapes=uniform_shapes(encoder_features,decoder_features)
    if set(draws)!=set(shapes):raise ValueError('Complete canonical initializer draws required')
    device=draws['encoder_0'].device
    for name,shape in shapes.items():
        value=draws[name]
        if (value.dtype!=torch.float32 or value.device!=device or value.shape!=shape
            or not torch.isfinite(value).all() or (value<0).any() or (value>=1).any()):
            raise ValueError('Finite matching float32 unit-uniform draws required')
    blocks=[draws[f'encoder_{i}']*.1-.05 for i in range(6)]
    result=dict(encoder_wi=torch.cat(blocks[:3]),encoder_wh=torch.cat(blocks[3:]),
                encoder_bi=torch.zeros(801,device=device),encoder_bh=torch.zeros(801,device=device),
                decoder_gate_bias=torch.ones(534,device=device),decoder_candidate_bias=torch.zeros(267,device=device),
                projection_bias=torch.zeros(1,device=device))
    for name in ['decoder_gate_kernel','decoder_candidate_kernel','projection_kernel']:
        fan_in,fan_out=shapes[name];limit=math.sqrt(6/(fan_in+fan_out))
        result[name]=draws[name]*(2*limit)-limit
    return result
