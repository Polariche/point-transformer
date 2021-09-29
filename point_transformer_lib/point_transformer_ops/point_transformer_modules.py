from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import point_transformer_ops.point_transformer_utils as pt_utils

import knn_cuda

class knn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k, return_ind=False):
        shp = x.shape

        x = x.view(-1, *x.shape[-2:])
        y = y.view(-1, *y.shape[-2:])

        dist = torch.zeros((x.shape[-3], x.shape[-2], y.shape[-2]), dtype=x.dtype, device=x.device)
        ind = torch.zeros((x.shape[-3], x.shape[-2], y.shape[-2]), device=x.device).long()

        for i, (x_, y_) in enumerate(zip(x,y)):
            dist[i], ind[i] = knn_cuda.forward(x_,y_,k)

        ctx.save_for_backward(x, y, dist, ind, torch.tensor(shp))

        dist = dist.view((*shp[:-2], x.shape[-2], y.shape[-2]))
        ind = ind.view((*shp[:-2], x.shape[-2], y.shape[-2]))

        if return_ind:
            return dist, ind
        else:
            return dist

    @staticmethod
    def backward(ctx, grad_output):
        x, y, dist, ind, shp = ctx.saved_tensors

        d_x = torch.zeros_like(x)
        d_y = torch.zeros_like(y)

        for i, (x_, y_, dist_, ind_) in enumerate(zip(x,y, dist, ind)):
            d_x[i], d_y[i] = knn_cuda.backward(grad_output, x_, y_, dist_, ind_)

        d_x = d_x.view((*shp[:-2], *x.shape[-2:]))
        d_y = d_y.view((*shp[:-2], *y.shape[-2:]))

        return d_x, d_y, None

knn_f = knn.apply

def idx_pt(pts, idx):
    raw_size  = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(pts, 1, idx[..., None].expand(-1, -1, pts.size(-1)))
    return res.reshape(*raw_size,-1)


class PointTransformerBlock(nn.Module):
    def __init__(self, dim, k, pos_dim=3):
        super().__init__()
        self.k = k

        self.lin_q = nn.Linear(dim, dim, bias=False)
        self.lin_k = nn.Linear(dim, dim, bias=False)
        self.lin_v = nn.Linear(dim, dim, bias=False)

        self.mid = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.ReLU(), 
            nn.Linear(dim, dim))

        self.pos_enc = nn.Sequential(
            nn.Linear(pos_dim, dim), 
            nn.ReLU(), 
            nn.Linear(dim, dim))
    
        self.final = nn.Linear(dim, dim)

    def forward(self, x, pos):

        _, ind = knn_f(pos, pos, self.k, True)
        ind = ind.long()

        x_k = x[ind]                # (n, k, in_dim)
        x_ = x.unsqueeze(-2)        # (n, 1, in_dim)

        pos_k = pos[ind]            # (n, k, 3)
        pos_ = pos.unsqueeze(-2)    # (n, 1, 3)

        pe = self.pos_enc(pos_ - pos_k)

        y = self.mid(self.lin_q(x_) - self.lin_k(x_k) + pe)
        y = F.softmax(y / np.sqrt(y.shape[-1]), dim=-2)

        z = self.lin_v(x_k) + pe

        final = (y * z).sum(dim=-2)
        final = self.final(final) + x

        return final


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, k, sampling_ratio, fast=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.sampling_ratio = sampling_ratio
        self.fast = fast
        self.mlp = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, x, p1):
        """
        inputs
            x: (B, N, in_channels) shaped torch Tensor (A set of feature vectors)
            p1: (B, N, 3) shaped torch Tensor (3D coordinates)
        outputs
            y: (B, M, out_channels) shaped torch Tensor
            p2: (B, M, 3) shaped torch Tensor
        M = N * sampling ratio
        """
        B, N, _ = x.shape
        M = int(N * self.sampling_ratio)

        # 1: Farthest Point Sampling
        p1_flipped = p1.transpose(1, 2).contiguous()
        p2 = (
            pt_utils.gather_operation(
                p1_flipped, pt_utils.farthest_point_sample(p1, M)
            )
            .transpose(1, 2)
            .contiguous()
        )  # p2: (B, M, 3)

        # 2: kNN & MLP
        knn_fn = pt_utils.kNN_torch if self.fast else pt_utils.kNN
        neighbors = knn_fn(p2, p1, self.k)  # neighbors: (B, M, k)

        # 2-1: Apply MLP onto each feature
        x_flipped = x.transpose(1, 2).contiguous()
        mlp_x = (
            self.mlp(x_flipped).transpose(1, 2).contiguous()
        )  # mlp_x: (B, N, out_channels)

        # 2-2: Extract features based on neighbors
        features = pt_utils.index_points(mlp_x, neighbors)  # features: (B, M, k, out_channels)

        # 3: Local Max Pooling
        y = torch.max(features, dim=2)[0]  # y: (B, M, out_channels)

        return y, p2


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )
        self.lateral_mlp = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x1, p1, x2, p2):
        """
            x1: (B, N, in_channels) torch.Tensor
            p1: (B, N, 3) torch.Tensor
            x2: (B, M, out_channels) torch.Tensor
            p2: (B, M, 3) torch.Tensor
        Note that N is smaller than M because this module upsamples features.
        """
        x1 = self.up_mlp(x1.transpose(1, 2).contiguous())
        dist, idx= pt_utils.three_nn(p2, p1)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pt_utils.three_interpolate(
            x1, idx, weight
        )
        x2 = self.lateral_mlp(x2.transpose(1, 2).contiguous())
        y = interpolated_feats + x2
        return y.transpose(1, 2).contiguous(), p2