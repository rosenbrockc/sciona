"""Tensor-only model warm-start checkpoints; no dataset or optimizer metadata."""
import io
import torch


def checkpoint_bytes(model):
    state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    stream=io.BytesIO()
    torch.save(state,stream)
    return stream.getvalue()


def restore_checkpoint(model,payload):
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    expected=model.state_dict()
    if not isinstance(state,dict) or state.keys()!=expected.keys():
        raise ValueError('Exact model state keys required')
    for name,value in state.items():
        target=expected[name]
        if not isinstance(value,torch.Tensor) or value.shape!=target.shape or value.dtype!=target.dtype or value.layout!=torch.strided or not torch.isfinite(value).all():
            raise ValueError('Invalid model state tensor')
    model.load_state_dict(state,strict=True)

