"""CHAMPS subtype unscaling, requested-row alignment and source precision.

The MIT Bosch predictor serializes each model's output to six decimal places
before ensemble reduction. Preserve that boundary without intermediate CSVs.
Caller-owned opaque keys are mapped separately from float32 source tensor IDs.
"""
from collections.abc import Mapping
import math

import torch


def predict_batches(model,batches,id_to_key: Mapping[int,str]) -> dict[str,float]:
    if not isinstance(id_to_key,Mapping):
        raise ValueError('CHAMPS prediction requires an ID mapping')
    if any(isinstance(k,bool) or not isinstance(k,int) or not 0<=k<2**24 for k in id_to_key):
        raise ValueError('CHAMPS internal IDs must be exactly representable nonnegative float32 integers')
    if any(not isinstance(k,str) or not k for k in id_to_key.values()) or len(set(id_to_key.values()))!=len(id_to_key):
        raise ValueError('CHAMPS output keys must be unique nonempty strings')
    was_training=model.training
    output={}
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                if len(batch)!=10:
                    raise ValueError('CHAMPS prediction requires ten packed tensors')
                _,atoms,positions,bonds,distances,triplets,angles,quads,dihedrals,targets=batch
                bonds,distances,targets=bonds[:,:406],distances[:,:406],targets[:,:406]
                prediction,_=model(atoms,positions,bonds,distances,triplets,angles,quads,dihedrals)
                if prediction.shape!=(atoms.shape[0],68,bonds.shape[1]) or not torch.isfinite(prediction).all():
                    raise ValueError('CHAMPS model returned invalid predictions')
                padded=torch.cat([prediction.new_zeros(prediction.shape[0],1,prediction.shape[2]),prediction],dim=1)
                unscaled=padded.gather(1,bonds[:,:,1][:,None,:])[:,0,:]*targets[:,:,2]+targets[:,:,1]
                selected=(bonds[:,:,0]>0)&(targets[:,:,3]>0)
                ids=targets[:,:,0][selected].cpu().tolist()
                values=unscaled[selected].cpu().tolist()
                for raw_id,value in zip(ids,values):
                    if not math.isfinite(raw_id) or raw_id!=int(raw_id) or int(raw_id) not in id_to_key:
                        raise ValueError('CHAMPS prediction contains an unknown internal ID')
                    key=id_to_key[int(raw_id)]
                    if key in output or not math.isfinite(value):
                        raise ValueError('CHAMPS prediction contains duplicate IDs or nonfinite values')
                    output[key]=float(format(value,'.6f'))
    finally:
        model.train(was_training)
    if set(output)!=set(id_to_key.values()):
        raise ValueError('CHAMPS model did not predict every requested key')
    return {key:output[key] for key in id_to_key.values()}
