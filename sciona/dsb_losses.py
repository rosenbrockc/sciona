"""Candidate differentiable losses for the source-pinned competition graph.

Source: lfz/DSB2017, commit 0ac3eb9f383bf0127c587e0502c59ac84d9ba6a2.
MIT notice: docs/licenses/DSB2017-MIT.txt. These are not catalog approvals.
"""
import numbers
import torch
from torch.nn import functional as F


class LearnedNoisyOR(torch.nn.Module):
    """Source trainable dummy-proposal baseline, initialized at -30."""
    def __init__(self):
        super().__init__()
        self.baseline = torch.nn.Parameter(torch.tensor([-30.], dtype=torch.float32))

    def forward(self, probabilities):
        _probabilities(probabilities)
        if probabilities.device != self.baseline.device or probabilities.dtype != self.baseline.dtype:
            raise ValueError('probabilities and learned baseline must share dtype and device')
        base_prob = torch.sigmoid(self.baseline)
        return 1 - torch.prod(1 - probabilities, dim=1) * (1 - base_prob.expand(probabilities.size(0)))


def _probabilities(values):
    if (not isinstance(values, torch.Tensor) or values.ndim != 2 or min(values.shape) < 1
            or not values.is_floating_point() or not bool(torch.isfinite(values).all())
            or not bool(((values >= 0) & (values <= 1)).all())):
        raise ValueError('proposal probabilities must be a nonempty finite B,K tensor in [0,1]')


def classifier_loss(case_probability, proposal_probability, case_labels, known_proposals,
                    miss_threshold=.03, miss_ratio=1.):
    """Source case BCE plus batch/proposal-averaged log(p+0.001) miss penalty."""
    _probabilities(proposal_probability)
    batch, count = proposal_probability.shape
    for values, shape in [(case_probability, (batch,)), (case_labels, (batch,)),
                          (known_proposals, (batch, count))]:
        if (not isinstance(values, torch.Tensor) or values.shape != shape
                or values.dtype != proposal_probability.dtype or values.device != proposal_probability.device
                or not bool(torch.isfinite(values).all()) or not bool(((values >= 0) & (values <= 1)).all())):
            raise ValueError('classifier probabilities, labels, and proposal indicators must have matching contracts')
    if not 0 < miss_threshold <= 1 or not 0 <= miss_ratio < float('inf'):
        raise ValueError('invalid miss threshold or weight')
    classification = F.binary_cross_entropy(case_probability, case_labels)
    # Source explicitly casts the nondifferentiable indicator to float32.
    mask = (proposal_probability < miss_threshold).float()
    miss = -torch.sum(mask * known_proposals * torch.log(proposal_probability + .001)) / batch / count
    return dict(total=classification + miss_ratio * miss, classification=classification, miss=miss)


def detector_loss(output, labels, num_hard=2, training=True):
    """Source sigmoid/BCE plus four SmoothL1 means and batch-scaled hard mining.

    Empty positive sets retain the source half-weight negative loss. Empty
    negative sets are rejected: the source otherwise produces an undefined mean.
    The returned losses preserve gradients; counts are diagnostic integers.
    """
    if (not isinstance(output, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or output.shape != labels.shape or output.ndim < 3 or output.shape[-1] != 5
            or output.shape[0] < 1 or not output.is_floating_point() or not labels.is_floating_point()
            or output.device != labels.device or output.dtype != labels.dtype
            or not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(labels).all())):
        raise ValueError('output and labels must be matching finite floating tensors ending in five values')
    if isinstance(num_hard, bool) or not isinstance(num_hard, numbers.Integral) or num_hard < 0:
        raise ValueError('num_hard must be a nonnegative integer')
    batch_size = output.shape[0]
    flat, truth = output.reshape(-1, 5), labels.reshape(-1, 5)
    states = truth[:, 0]
    if not bool(((states == -1) | (states == 0) | (states == 1)).all()):
        raise ValueError('anchor labels must be -1, 0, or 1')
    positive, negative = states > .5, states < -.5
    negative_indices = torch.where(negative)[0]
    if not negative_indices.numel():
        raise ValueError('source classification mean requires at least one negative anchor')
    if num_hard and training:
        indices = torch.topk(flat[negative_indices, 0], min(num_hard * batch_size, len(negative_indices))).indices
        negative_indices = negative_indices[indices]
    negative_probability = torch.sigmoid(flat[negative_indices, 0])
    classification = .5 * F.binary_cross_entropy(negative_probability, truth[negative_indices, 0] + 1)
    if bool(positive.any()):
        positive_probability = torch.sigmoid(flat[positive, 0])
        classification = classification + .5 * F.binary_cross_entropy(positive_probability, states[positive])
        regression = torch.stack([F.smooth_l1_loss(flat[positive, i], truth[positive, i]) for i in range(1, 5)])
        positive_correct = int((positive_probability >= .5).sum())
    else:
        regression = flat[:0].sum() + flat.new_zeros(4)
        positive_correct = 0
    total = classification
    for value in regression:
        total = total + value
    return dict(total=total, classification=classification, regression=regression,
                positive_correct=positive_correct, positive_count=int(positive.sum()),
                negative_correct=int((negative_probability < .5).sum()),
                negative_count=len(negative_indices), negative_indices=negative_indices)
