"""Plain-output HuBMAP network with source batch-level inference shortcut."""
import math
from sciona.hubmap_network import UNET_SERESNEXT101


class InferenceNetwork(UNET_SERESNEXT101):
    def __init__(self,resolution=(320,320),*,classifier_threshold=.5,encoder_state=None):
        if isinstance(classifier_threshold,bool) or not isinstance(classifier_threshold,(int,float)) or not math.isfinite(classifier_threshold) or not 0<=classifier_threshold<=1:
            raise ValueError('Finite classifier probability threshold required')
        super().__init__(resolution,False,False,None,load_weights=False,encoder_state=encoder_state)
        # Base forward contains the original shortcut, previously disabled for
        # training. Enable it only for plain inference output: source auxiliary
        # shortcut outputs contain invalid integer placeholders.
        self.clf_threshold=classifier_threshold
        self.eval()

    def train(self,mode=True):
        if mode:
            raise ValueError('InferenceNetwork is evaluation-only')
        return super().train(False)
