"""Source-style model-only checkpoints without runtime args or source metadata."""
import io
import torch
from sciona.dsb_components import _integer


def checkpoint_bytes(classifier,epoch):
    """Serialize source epoch/state_dict keys only; caller controls private storage.

    Source resume recreates optimizers. This is a model warm restart, not an
    exact continuation of optimizer/RNG state. No arbitrary args are serialized.
    """
    epoch=_integer(epoch,'epoch',0)
    state={k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()}
    stream=io.BytesIO();torch.save(dict(epoch=epoch,state_dict=state),stream)
    return stream.getvalue()


def restore_checkpoint(classifier,payload,start_epoch=0):
    """Strict tensor-only model restore; validate full state before mutation."""
    start_epoch=_integer(start_epoch,'start_epoch',0)
    checkpoint=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict) or set(checkpoint)!={'epoch','state_dict'}:
        raise ValueError('expected metadata-free epoch/state_dict checkpoint')
    epoch=_integer(checkpoint['epoch'],'checkpoint epoch',0)
    state=checkpoint['state_dict'];expected=classifier.state_dict()
    if not isinstance(state,dict) or state.keys()!=expected.keys():
        raise ValueError('checkpoint state keys differ')
    for key,value in state.items():
        if (not isinstance(value,torch.Tensor) or value.shape!=expected[key].shape
                or value.dtype!=expected[key].dtype or value.layout!=torch.strided
                or not bool(torch.isfinite(value).all())):
            raise ValueError('invalid checkpoint tensor contract')
    classifier.load_state_dict(state,strict=True)
    return start_epoch if start_epoch else epoch+1
