"""Source-order decoder dropout with explicit runtime masks.

Web Traffic model loop is MIT; see docs/licenses/WebTraffic-MIT.txt.
Mask streams are explicit: no TensorFlow random-sequence equivalence claimed.
"""
import torch
from .webtraffic_decoder import gru_block_step


def apply_mask(values, mask, keep_probability):
    if not 0 < keep_probability <= 1:
        raise ValueError('Keep probability must be in (0, 1]')
    if keep_probability == 1:
        return values
    if mask is None or mask.dtype != torch.bool or mask.shape != values.shape:
        raise ValueError('Explicit boolean mask matching tensor shape required')
    return values / keep_probability * mask.to(values.dtype)


def decode_with_masks(encoder_state, features, previous_y, cells, projection_kernel,
                      projection_bias, masks, keep_probabilities, attention=None):
    """Input dropout precedes the cell; state and output dropout use separate masks."""
    if len(encoder_state) != len(cells) or len(masks) != len(cells) or len(keep_probabilities) != len(cells):
        raise ValueError('State, weights and dropout specifications must match layers')
    states=tuple(encoder_state);previous=previous_y[:,None];targets=[];outputs=[]
    for day in range(features.shape[1]):
        parts=[previous,features[:,day]]
        if attention is not None:parts.append(attention[:,day])
        value=torch.cat(parts,dim=1);updated=[]
        for h,w,mask,keep in zip(states,cells,masks,keep_probabilities):
            def dropout(v,kind):
                selected=mask.get(kind)
                return apply_mask(v,None if selected is None else selected[day],keep[kind])
            cell_input=dropout(value,'input')
            raw=gru_block_step(cell_input,h,**w)
            updated.append(dropout(raw,'state'))
            value=dropout(raw,'output')
        states=tuple(updated)
        previous=value @ projection_kernel + projection_bias
        outputs.append(value);targets.append(previous.squeeze(-1))
    return torch.stack(targets),torch.stack(outputs),states
