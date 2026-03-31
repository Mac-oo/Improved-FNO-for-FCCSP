import torch
import numpy as np
import scipy.io
import torch.nn as nn
import operator
from functools import reduce

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MaskedUnitGaussianNormalizer(object):
    def __init__(self, x, mask, indices_to_normalize=None, eps=1e-6):
        super(MaskedUnitGaussianNormalizer, self).__init__()

        if indices_to_normalize is None:
            self.indices = list(range(x.shape[-1]))
        else:
            self.indices = indices_to_normalize

        self.eps = eps
        target_data = x[..., self.indices]                  # (..., C_sel)
        target_mask = mask[..., self.indices].float()       # (..., C_sel)
        stat_dim = tuple(range(x.ndim - 1))                 # the statistics of the dimension
        valid_count = torch.sum(target_mask, dim=stat_dim).clamp(min=1.0)
        self.mean = torch.sum(target_data * target_mask, dim=stat_dim) / valid_count

        mean_view = self.mean.view(*([1] * (x.ndim - 1)), -1)
        variance = torch.sum(((target_data - mean_view) ** 2) * target_mask, dim=stat_dim) / valid_count
        self.std = torch.sqrt(torch.clamp(variance, min=0.0) + self.eps)

    def encode(self, x, mask):
        x = x.clone()
        target = x[..., self.indices]
        target_mask = mask[..., self.indices].float()

        mean_view = self.mean.view(*([1] * (x.ndim - 1)), -1)
        std_view = self.std.view(*([1] * (x.ndim - 1)), -1)

        encoded = (target - mean_view) / std_view
        x[..., self.indices] = encoded * target_mask
        return x

    def decode(self, x, mask=None):
        x = x.clone()
        target = x[..., self.indices]

        mean_view = self.mean.view(*([1] * (x.ndim - 1)), -1)
        std_view = self.std.view(*([1] * (x.ndim - 1)), -1)
        decoded = target * std_view + mean_view

        if mask is not None:
            target_mask = mask[..., self.indices].float()
            decoded = decoded * target_mask

        x[..., self.indices] = decoded
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class UnitGaussianNormalizer(object):
    def __init__(self, x, indices_to_normalize, eps=1e-6):
        super(UnitGaussianNormalizer, self).__init__()
        self.indices = indices_to_normalize
        self.eps = eps
        target_data = x[..., self.indices]    # shape: (batch, x, y, z, len(indices))
        stat_dim = tuple(range(x.ndim - 1))    # batch, x, y, z

        self.mean = torch.mean(target_data, dim=stat_dim)
        self.std = torch.std(target_data, dim=stat_dim)

    def encode(self, x):
        x[..., self.indices] = (x[..., self.indices] - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x[..., self.indices] = (x[..., self.indices] * (self.std + self.eps)) + self.mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class MaskedMSELoss(nn.Module):
    def __init__(self, channel_average=True, eps=1e-8):
        super(MaskedMSELoss, self).__init__()
        self.channel_average = channel_average
        self.eps = eps

    def forward(self, pred, target, mask):
        mask = mask.float()
        diff2 = (pred - target) ** 2 * mask

        spatial_dims = tuple(range(1, pred.ndim - 1))   # (gx, gy, gz)
        num = diff2.sum(dim=spatial_dims)                # (B, C)
        den = mask.sum(dim=spatial_dims)                 # (B, C)

        valid = den > 0
        loss_bc = torch.zeros_like(num)
        loss_bc[valid] = num[valid] / den[valid].clamp(min=self.eps)
        # 每个样本对其有效通道平均
        valid_count_per_sample = valid.sum(dim=-1).clamp(min=1)
        loss_b = loss_bc.sum(dim=-1) / valid_count_per_sample   # (B,)

        if self.channel_average:
            return loss_b.mean()
        else:
            return loss_b.sum()


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c
