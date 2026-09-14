"""Pinned detector-to-classifier model initialization; no checkpoint IO paths."""
from sciona.dsb_checkpoint import restore_checkpoint,checkpoint_bytes
from sciona.dsb_components import _integer
from sciona.dsb_network import Net,CaseNet


def adapt_detector_checkpoint(detector_checkpoint,*,topk=5):
    """Transfer the detector state into shared features and initialize the case head.

    Matches training/classifier/adapt_ckpt.py: load detector, inject that instance
    into CaseNet, and emit epoch zero. Caller owns Torch initialization RNG state.
    The classifier head is newly initialized, not trained or prediction-approved.
    """
    topk=_integer(topk,'topk')
    detector=Net()
    restore_checkpoint(detector,detector_checkpoint)
    classifier=CaseNet(topk,nodulenet=detector)
    return checkpoint_bytes(classifier,0)
