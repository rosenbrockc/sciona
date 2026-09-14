"""Complete source ensembles and memory-bounded input differentiation.

Two passes evaluate fixed inference models: collect logits, then recompute each
model's vector-Jacobian product. No approximation of the ensemble objective.
"""
import torch
from sciona.adversarial_losses import ensemble_loss,infer_labels
from sciona.adversarial_inception_v3 import build_inception_v3
from sciona.adversarial_inception_v4 import build_inception_v4
from sciona.adversarial_inception_resnet import build_inception_resnet
from sciona.adversarial_resnet import build_resnet101
from sciona.adversarial_state_mapping import convert_state

ORDER={
    'non_targeted':('InceptionV3','AdvInceptionV3','Ens3AdvInceptionV3','Ens4AdvInceptionV3','InceptionV4','InceptionResnetV2','EnsAdvInceptionResnetV2','resnet_v2_101'),
    'targeted_large':('InceptionV3','AdvInceptionV3','Ens3AdvInceptionV3','Ens4AdvInceptionV3','EnsAdvInceptionResnetV2'),
    'targeted_small':('InceptionV3','EnsAdvInceptionResnetV2'),
}


def build_ensemble(branch, initializations):
    """Require explicit random or caller-decoded Slim state for every source scope."""
    if branch not in ORDER or not isinstance(initializations,dict) or initializations.keys()!=set(ORDER[branch]):
        raise ValueError('exact branch initialization scopes required')
    result=[]
    for scope in ORDER[branch]:
        spec=initializations[scope]
        if not isinstance(spec,dict) or spec.get('kind') not in ('random','slim_tensors'):
            raise ValueError('explicit initialization kind required')
        required={'kind','seed'} if spec['kind']=='random' else {'kind','seed','tensors'}
        if spec.keys()!=required:raise ValueError('exact initialization fields required')
        if scope=='resnet_v2_101':builder,family=build_resnet101,'resnet101'
        elif scope=='InceptionV4':builder,family=build_inception_v4,'inception_v4'
        elif 'Resnet' in scope:builder,family=build_inception_resnet,'inception_resnet'
        else:builder,family=build_inception_v3,'inception_v3'
        model=builder(seed=spec['seed'])
        if spec['kind']=='slim_tensors':
            model.load_state_dict(convert_state(model,spec['tensors'],family=family,scope=scope),strict=True)
        result.append(model.eval().requires_grad_(False))
    return tuple(result)


def gradient(models, x, *, branch, iteration, labels=None):
    """Return exact-objective loss, frozen/supplied labels and NCHW input gradient.

    Models must be fixed deterministic inference backbones built above. Sequential
    VJP accumulation can differ in floating reduction order from a single TF graph.
    """
    if branch not in ORDER or len(models)!=len(ORDER[branch]):raise ValueError('exact branch model count required')
    if type(iteration) is not int or iteration<0:raise ValueError('nonnegative iteration required')
    if (not isinstance(x,torch.Tensor) or x.dtype!=torch.float32 or x.device.type!='cpu'
            or x.ndim!=4 or tuple(x.shape[1:])!=(3,299,299) or x.shape[0]<1
            or not torch.isfinite(x).all() or torch.any(x < -1) or torch.any(x > 1)):
        raise ValueError('normalized finite full-resolution CPU float32 input required')
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in models):
        raise ValueError('frozen inference models required')
    main=[];aux=[];prob=[]
    with torch.no_grad():
        for i,model in enumerate(models):
            logits,ends=model(x)
            if logits.shape!=(len(x),1001):raise ValueError('source class count required')
            main.append(logits.detach().requires_grad_())
            prob.append(ends['predictions'] if 'predictions' in ends else ends['Predictions'])
            if branch!='non_targeted' or i!=7:aux.append(ends['AuxLogits'].detach().requires_grad_())
    if branch=='non_targeted':labels=infer_labels(prob,iteration=iteration,previous=labels)
    loss=ensemble_loss(main,aux,labels,branch=branch)
    derivatives=torch.autograd.grad(loss,main+aux)
    total=torch.zeros_like(x)
    for i,model in enumerate(models):
        local=x.detach().requires_grad_()
        logits,ends=model(local)
        if not torch.equal(logits.detach(),main[i].detach()):raise ValueError('model changed between inference passes')
        outputs=[logits];weights=[derivatives[i]]
        if i<len(aux):
            if not torch.equal(ends['AuxLogits'].detach(),aux[i].detach()):raise ValueError('auxiliary model changed between passes')
            outputs.append(ends['AuxLogits']);weights.append(derivatives[len(main)+i])
        contribution,=torch.autograd.grad(outputs,local,grad_outputs=weights)
        total=total+contribution
    if not torch.isfinite(total).all():raise ValueError('nonfinite ensemble input gradient')
    return {'loss':float(loss.detach()),'labels':labels.detach().clone(),'gradient':total.detach()}
