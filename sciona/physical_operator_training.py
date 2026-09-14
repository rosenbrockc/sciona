"""Full-batch CPU Fourier-operator training with held-out checkpoint selection."""
from dataclasses import dataclass
import torch
from sciona.physical_operator_network import FourierOperator1d
from sciona.physical_operator_state import prepare,tensors

@dataclass
class FittedOperator:
    model: object
    history: list
    best_epoch: int
    validation_mse: float


def fit(payload):
    p=prepare(payload);c=p['controls']
    x,t=tensors(p['training'],p['length']);vx,vt=tensors(p['validation'],p['length'])
    y=torch.tensor(p['training']['targets'],dtype=torch.float64)
    vy=torch.tensor(p['validation']['targets'],dtype=torch.float64)
    # Initialization does not alter the caller's CPU random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        model=FourierOperator1d(c['width'],c['modes'],c['depth']).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    history=[];best=float('inf');best_epoch=0;checkpoint=None
    for epoch in range(c['epochs']+1):
        model.eval()
        with torch.no_grad():
            training_mse=float((model(x,t)-y).square().mean())
            validation_mse=float((model(vx,vt)-vy).square().mean())
        if not torch.isfinite(torch.tensor([training_mse,validation_mse],dtype=torch.float64)).all():raise ValueError('Nonfinite training trajectory')
        history.append(dict(epoch=epoch,training_mse=training_mse,validation_mse=validation_mse))
        if validation_mse<best:
            best=validation_mse;best_epoch=epoch
            checkpoint={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch==c['epochs']:break
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=(model(x,t)-y).square().mean();loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Invalid training gradients')
        optimizer.step()
    model.load_state_dict(checkpoint);model.eval()
    return FittedOperator(model,history,best_epoch,best)
