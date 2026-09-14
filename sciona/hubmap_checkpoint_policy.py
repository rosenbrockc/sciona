"""Source epoch checkpoint/early-stop decisions, without filesystem side effects."""
from dataclasses import dataclass
import math


@dataclass
class CheckpointPolicy:
    best_loss: float = 1e99
    score_at_best_loss: float = -1e99
    best_score: float = -1e99
    best_epoch: int = 0
    stale_epochs: int = 0

    def decide(self, *, epoch, validation_loss, validation_score, early_stopping, patience, snapshot_period=19):
        if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<1:
            raise ValueError('Positive integer epoch required')
        if not math.isfinite(validation_loss) or not math.isfinite(validation_score):
            raise ValueError('Finite validation statistics required')
        if not isinstance(early_stopping,bool) or isinstance(patience,bool) or not isinstance(patience,int) or patience<0:
            raise ValueError('Boolean early stopping and nonnegative patience required')
        if isinstance(snapshot_period,bool) or not isinstance(snapshot_period,int) or snapshot_period<2:
            raise ValueError('Snapshot period at least two required')
        saves=[]
        if early_stopping:
            if validation_loss<self.best_loss:
                self.score_at_best_loss=validation_score
                self.best_loss=validation_loss
                self.best_epoch=epoch
                self.stale_epochs=0
                saves.append('best_loss')
            else:
                self.stale_epochs+=1
            if self.stale_epochs>patience:
                return dict(save=saves,stop=True,step_scheduler=False)
        else:
            saves.append('best_loss')
        if validation_score>self.best_score:
            self.best_score=validation_score
            saves.append('best_score')
        if any(epoch%divisor==0 for divisor in [snapshot_period+1,snapshot_period,snapshot_period-1]):
            saves.append('snapshot')
        return dict(save=saves,stop=False,step_scheduler=True)
