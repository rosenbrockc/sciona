"""Independent implementation of the recovered grouped-attention architecture.

Pending full Santander lifecycle reconstruction; not an approved provider.
Constant intercepts use one embedding entry because feature identities are not
inputs. Initialization uses PyTorch defaults and is not historical seed replay.
"""
import torch
from torch import nn


class GroupedAttentionClassifier(nn.Module):
    def __init__(self, groups):
        super().__init__()
        if type(groups) is not int or groups < 1:
            raise ValueError('Expected positive integer feature-group count')
        self.groups = groups
        self.category = nn.Embedding(6, 12)
        self.value_intercept = nn.Embedding(1, 2)
        self.weight_intercept = nn.Embedding(1, 2)
        self.value_intercept_norm = nn.BatchNorm1d(groups)
        self.weight_intercept_norm = nn.BatchNorm1d(groups)
        self.raw_branch = nn.Sequential(nn.Linear(3, 50), nn.ReLU(), nn.Linear(50, 10))
        self.substituted_branch = nn.Sequential(nn.Linear(3, 50), nn.ReLU(), nn.Linear(50, 10))
        self.category_norm = nn.BatchNorm1d(groups)
        self.raw_norm = nn.BatchNorm1d(groups)
        self.substituted_norm = nn.BatchNorm1d(groups)
        self.branch_dropout = nn.Dropout(0.08)
        self.weight_norm = nn.BatchNorm1d(34)
        self.weight_network = nn.Sequential(nn.Linear(34, 5), nn.ReLU(), nn.Linear(5, 1))
        self.pooled_norm = nn.BatchNorm1d(32)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, categories, raw, substituted, *, return_weights=False):
        inputs = (categories, raw, substituted)
        if not all(isinstance(x, torch.Tensor) for x in inputs):
            raise ValueError('Expected tensor inputs')
        if categories.ndim != 2 or categories.shape[0] < 1 or categories.shape[1] != self.groups:
            raise ValueError('Expected nonempty aligned feature groups')
        if raw.shape != categories.shape or substituted.shape != categories.shape:
            raise ValueError('Feature group shapes differ')
        if categories.dtype != torch.long or torch.any(categories < 0) or torch.any(categories > 5):
            raise ValueError('Expected six-token integer categories')
        parameter = self.category.weight
        if any(x.device != parameter.device for x in inputs):
            raise ValueError('Inputs and network must share device')
        if any(x.dtype != parameter.dtype or not torch.isfinite(x).all() for x in (raw, substituted)):
            raise ValueError('Expected finite continuous tensors matching network dtype')
        if self.training and raw.shape[0] < 2:
            raise ValueError('Training requires at least two rows for batch normalization')
        if type(return_weights) is not bool:
            raise ValueError('Expected boolean weight-output control')
        batch = raw.shape[0]
        token = torch.zeros((batch, self.groups), dtype=torch.long, device=raw.device)
        intercept = self.value_intercept_norm(self.value_intercept(token))
        categorical = self.branch_dropout(self.category_norm(self.category(categories)))
        original = self.branch_dropout(self.raw_norm(
            self.raw_branch(torch.cat((raw.unsqueeze(-1), intercept), dim=-1))))
        replacement = self.branch_dropout(self.substituted_norm(
            self.substituted_branch(torch.cat((substituted.unsqueeze(-1), intercept), dim=-1))))
        weight_intercept = self.weight_intercept_norm(self.weight_intercept(token))
        joined = torch.cat((categorical, weight_intercept, original, replacement), dim=-1)
        scores = self.weight_network(self.weight_norm(joined.reshape(-1, 34))).reshape(batch, self.groups)
        weights = scores.softmax(dim=1)
        pooled = torch.cat([(part * weights.unsqueeze(-1)).sum(dim=1)
            for part in (categorical, original, replacement)], dim=-1)
        logits = self.head(self.pooled_norm(pooled))
        return (logits, weights) if return_weights else logits
