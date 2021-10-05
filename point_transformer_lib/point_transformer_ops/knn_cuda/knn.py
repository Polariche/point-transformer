import torch
import knn_cuda

import sys
import os

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import pmath

class knn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k):
        dist, ind = knn_cuda.forward(x,y,k)
        ctx.save_for_backward(x, y, dist, ind)

        return dist

    @staticmethod
    def backward(ctx, grad_output):
        x, y, dist, ind = ctx.saved_tensors

        d_x, d_y = knn_cuda.backward(grad_output, x, y, dist, ind)
        return d_x, d_y, None


class hyper_knn_test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k, curv):
        dist = knn_cuda.hyper_forward(x,y,k, curv)

        return dist


if __name__ == "__main__":
    x = torch.randn(33,2).cuda().requires_grad_()
    y = torch.randn(33,2).cuda().requires_grad_()
    k = 33

    #print(pmath.dist(x,y,c=1.0))
    print(torch.pow(x.unsqueeze(1) - y.unsqueeze(0), 2).sum(dim=-1))
    print(knn_cuda.hyper_forward(x,y,k,1.0))
    



