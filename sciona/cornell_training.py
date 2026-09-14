"""Full-length Cornell training phase with explicit checkpoint selection."""
from pathlib import Path
import numpy as np
import torch
from sciona.cornell_schedule import learning_rate
from sciona.cornell_validation import clip_f1


def checked_batch(batch,training):
    waveform,labels,secondary=batch
    values=torch.as_tensor(waveform,dtype=torch.float32)
    truth=torch.as_tensor(labels,dtype=torch.float32)
    secondary=torch.as_tensor(secondary,dtype=torch.float32)
    clips=1 if training else 2
    if values.ndim!=3 or values.shape[1:]!=(clips,960000) or not len(values):
        raise ValueError('Expected complete thirty-second mono clips')
    if truth.shape!=(len(values),clips,264) or secondary.shape!=truth.shape:
        raise ValueError('Expected264class labels for every clip')
    if not torch.isfinite(values).all() or not torch.isfinite(truth).all() or not torch.isfinite(secondary).all():
        raise ValueError('Nonfinite training values')
    if not torch.all((truth==0)|(truth==1)) or not torch.all((secondary==0)|(secondary==1)) or torch.any(secondary>truth):
        raise ValueError('Expected binary labels with secondary subset')
    return values,truth,secondary


def train_phase(source,model,*,training_batches,validation_batches,steps_per_epoch,epochs,
                peak,mixup,augmented_loss,seed,checkpoint_directory,select_epoch=None):
    """Run one source phase; batch factories supply a full population each epoch.

    Checkpoint selection is explicit: a retained epoch or best score, latest tie.
    Original hand-picked competition checkpoints are not claimed equivalent.
    The source two-epoch validation cadence and five-checkpoint retention apply.
    Model-only continuation calls this function anew with restored model weights.
    """
    if type(epochs) is not int or epochs<2 or type(mixup) is not bool:
        raise ValueError('At least two epochs and explicit mixup required')
    learning_rate(0,steps_per_epoch=steps_per_epoch,epochs=epochs,peak=peak)
    directory=Path(checkpoint_directory);directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise ValueError('Checkpoint directory must be empty')
    opt=torch.optim.AdamW(model.parameters(),lr=peak,betas=(.9,.999),eps=1e-8,weight_decay=.0001,amsgrad=True)
    criterion=source.loss(augmented=augmented_loss)
    rng=np.random.RandomState(seed);retained=[];reports=[];iteration=0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch in range(1,epochs+1):
            model.train();losses=[]
            for batch in training_batches(epoch):
                if len(losses)>=steps_per_epoch:raise ValueError('Too many training batches')
                values,labels,secondary=checked_batch(batch,True)
                lam=None
                if mixup:
                    if len(values)%2:raise ValueError('Mixup requires paired examples')
                    weights=[]
                    for _ in range(len(values)//2):
                        value=rng.beta(1.,1.);weights.extend((value,1-value))
                    lam=torch.tensor(weights,dtype=torch.float32)
                    labels=source.mix(labels,lam);secondary=source.mix(secondary,lam)
                for group in opt.param_groups:group['lr']=learning_rate(iteration,steps_per_epoch=steps_per_epoch,epochs=epochs,peak=peak)
                opt.zero_grad();output=model((values,lam))
                loss,_=criterion(output['clipwise_output'],dict(all_labels=labels,secondary_labels=secondary))
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise ValueError('Nonfinite training gradient')
                opt.step();losses.append(float(loss.detach()));iteration+=1
                del output,loss,values,labels,secondary
            if len(losses)!=steps_per_epoch:raise ValueError('Incomplete training epoch')
            report=dict(epoch=epoch,mean_loss=float(np.mean(losses)),iterations=iteration)
            if epoch%2==0:
                model.eval();scores=[];counts=[];validation_losses=[]
                with torch.no_grad():
                    for batch in validation_batches():
                        values,labels,secondary=checked_batch(batch,False)
                        prediction=model((values,None))['clipwise_output']
                        validation_loss,_=criterion(prediction,dict(all_labels=labels,secondary_labels=secondary))
                        if not torch.isfinite(validation_loss):raise ValueError('Nonfinite validation loss')
                        validation_losses.append(float(validation_loss))
                        scores.append(clip_f1(prediction.cpu().numpy(),labels.numpy()));counts.append(len(values))
                if not counts:raise ValueError('Empty validation population')
                score=float(np.average(scores,weights=counts));report['validation_f1']=score
                report['validation_loss']=float(np.average(validation_losses,weights=counts))
                if len(retained)<5 or score>retained[0][0]:
                    path=directory/f'epoch-{epoch}.pt'
                    torch.save(dict(format='cornell-phase.v1',model=model.state_dict(),optimizer=opt.state_dict(),
                        epoch=epoch,iteration=iteration,score=score,torch_rng=torch.random.get_rng_state(),
                        mixup_rng=rng.get_state(),schedule=dict(steps_per_epoch=steps_per_epoch,epochs=epochs,peak=peak)),path)
                    retained.append((score,epoch,path));retained.sort(key=lambda x:x[0])
                    if len(retained)>5:retained.pop(0)[2].unlink()
            reports.append(report)
    selected=next((r for r in retained if r[1]==select_epoch),None) if select_epoch is not None else retained[-1]
    if selected is None:raise ValueError('Requested epoch was not retained')
    # Only this process's own freshly written checkpoint is read here.
    state=torch.load(selected[2],map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    return dict(reports=reports,selected_epoch=selected[1],selected_score=selected[0],
                checkpoint=selected[2],retained_epochs=[r[1] for r in retained])
