"""Neural CV lifecycle for the pending source-compared Santander realization.

Global preparation is intentional and precludes unbiased CV claims. This is
only the neural branch; pseudo-label retraining and tree blending are separate.
"""
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from sciona.santander_preparation import prepare_neural_populations
from sciona.santander_augmentation import shuffle_feature_groups
from sciona.santander_network import GroupedAttentionClassifier


def cycle_parameters(step, total, maximum):
    """Legacy two-phase cosine values used before the indexed optimizer step."""
    warm = int(total * .3)
    if total < 4 or not 0 <= step < total:
        raise ValueError('Invalid one-cycle step range')
    if step <= warm:
        fraction = step / warm
        low, high, beta_start, beta_end = maximum/25, maximum, .95, .85
    else:
        fraction = (step-warm)/(total-warm)
        low, high, beta_start, beta_end = maximum, maximum/250000, .85, .95
    ratio = (1-math.cos(math.pi*fraction))/2
    return low+(high-low)*ratio, beta_start+(beta_end-beta_start)*ratio


def _batches(order, size):
    batches = [order[i:i+size] for i in range(0, len(order), size)]
    if len(batches[-1]) == 1:
        tail = batches.pop()
        batches[-1] = np.concatenate((batches[-1], tail))
    return batches


def train_neural_cv(training, labels, test, *, folds=10, seeds=(42,),
                    fold_seed=42, epochs=15, batch_size=1024, maximum_lr=.01):
    """Train every fold/seed, restore best AUROC epoch and average query ranks.

Uses CPU, repeated epoch permutation twice, per-batch triplet augmentation,
Adam beta2=.99 with decoupled decay=.1 including normalization parameters, and
legacy cosine scheduling. Singleton tails merge into the preceding batch.
NumPy augmentation RNG and PyTorch initialization differ from historical seed
replay. Query ranks are scores, not calibrated probabilities. Caller controls
allow bounded synthetic validation of the complete branch lifecycle.
"""
    for value, minimum in ((folds,2),(epochs,4),(batch_size,2),(fold_seed,0)):
        if type(value) is not int or value < minimum:
            raise ValueError('Invalid integer training control')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0<=s<2**32 for s in seeds) or fold_seed>=2**32:
        raise ValueError('Expected distinct bounded integer seeds')
    if type(maximum_lr) not in (int,float) or not math.isfinite(maximum_lr) or not 0<maximum_lr<=1:
        raise ValueError('Invalid maximum learning rate')
    prepared = prepare_neural_populations(training, labels, test)
    y = prepared['labels']
    if np.bincount(y, minlength=2).min() < folds:
        raise ValueError('Each class must supply every stratified fold')
    train_values = [prepared['training_'+k] for k in ('categories','raw','substituted')]
    test_values = [torch.from_numpy(prepared['test_'+k]) for k in ('categories','raw','substituted')]
    predictions, records = [], []
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=fold_seed).split(y,y))
    with torch.random.fork_rng(devices=[]):
        for fold, (fit, valid) in enumerate(splits):
            validation = [torch.from_numpy(a[valid]) for a in train_values]
            steps_per_epoch = len(_batches(np.tile(fit,2),batch_size))
            total = steps_per_epoch * epochs
            for seed in seeds:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = GroupedAttentionClassifier(train_values[0].shape[1])
                optimizer = torch.optim.Adam(model.parameters(), lr=maximum_lr/25, betas=(.95,.99), weight_decay=0)
                best, saved, best_epoch, history, step = -math.inf, None, None, [], 0
                for epoch in range(epochs):
                    model.train()
                    order = np.tile(rng.permutation(fit),2)
                    for indices in _batches(order,batch_size):
                        arrays = shuffle_feature_groups(*(a[indices] for a in train_values),y[indices],rng=rng,training=True)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(*(torch.from_numpy(a) for a in arrays[:3]))
                        loss = torch.nn.functional.cross_entropy(logits,torch.from_numpy(arrays[3]))
                        if not torch.isfinite(loss): raise ValueError('Nonfinite neural training loss')
                        loss.backward()
                        lr, beta = cycle_parameters(step,total,maximum_lr)
                        for group in optimizer.param_groups: group.update(lr=lr,betas=(beta,.99))
                        with torch.no_grad():
                            for parameter in model.parameters():
                                if parameter.grad is not None:
                                    if not torch.isfinite(parameter.grad).all(): raise ValueError('Nonfinite gradient')
                                    parameter.mul_(1-.1*lr)
                        optimizer.step()
                        step += 1
                    model.eval()
                    with torch.no_grad(): probabilities=model(*validation).softmax(1)[:,1].numpy()
                    auc=float(roc_auc_score(y[valid],probabilities))
                    history.append(auc)
                    if auc>best:
                        best,best_epoch=auc,epoch
                        saved={k:v.detach().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(saved)
                model.eval()
                with torch.no_grad():
                    restored=model(*validation).softmax(1)[:,1].numpy()
                    predicted=model(*test_values).softmax(1)[:,1].numpy()
                if float(roc_auc_score(y[valid],restored)) != best or not np.isfinite(predicted).all():
                    raise ValueError('Best checkpoint verification failed')
                predictions.append(predicted)
                records.append(dict(fold=fold,seed=seed,fit_rows=len(fit),validation_rows=len(valid),
                    optimizer_steps=step,best_epoch=best_epoch,validation_auc=history,restored_auc=best))
    mean_ranks=np.mean([rankdata(p,method='average') for p in predictions],axis=0)
    return dict(mean_ranks=mean_ranks,model_probabilities=np.stack(predictions),models=records,
        validation_scope='Globally encoded and scaled; not fold-isolated validation')
