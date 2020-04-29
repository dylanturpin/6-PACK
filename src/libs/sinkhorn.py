import math
import time
import torch
import torch.utils.cpp_extension
import os

src_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sinkhorn.cu')
wasserstein_ext = torch.utils.cpp_extension.load(
    name="wasserstein",
    sources=[src_path],
    extra_cuda_cflags=["--expt-relaxed-constexpr"]
)
def sinkstep(dist, log_nu, log_u, lam: float):
    # dispatch to optimized GPU implementation for GPU tensors, slow fallback for CPU
    if dist.is_cuda:
        return wasserstein_ext.sinkstep(dist, log_nu, log_u, lam)
    assert dist.dim() == 2 and log_nu.dim() == 2 and log_u.dim() == 2
    assert dist.size(0) == log_u.size(1) and dist.size(1) == log_nu.size(1) and log_u.size(0) == log_nu.size(0)
    log_v = log_nu.clone()
    for b in range(log_u.size(0)):
        log_v[b] -= torch.logsumexp(-dist/lam+log_u[b, :, None], 1)
    return log_v


class SinkhornOT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, nu, dist, lam=1e-3, N=100):
        assert mu.dim() == 2 and nu.dim() == 2 and dist.dim() == 2
        bs = mu.size(0)
        d1, d2 = dist.size()
        assert nu.size(0) == bs and mu.size(1) == d1 and nu.size(1) == d2
        log_mu = mu.log()
        log_nu = nu.log()
        log_u = torch.full_like(mu, -math.log(d1))
        log_v = torch.full_like(nu, -math.log(d2))
        for i in range(N):
            log_v = sinkstep(dist, log_nu, log_u, lam)
            log_u = sinkstep(dist.t(), log_mu, log_v, lam)

        # this is slight abuse of the function. it computes (diag(exp(log_u))*Mt*exp(-Mt/lam)*diag(exp(log_v))).sum()
        # in an efficient (i.e. no bxnxm tensors) way in log space
        distances = (-sinkstep(-dist.log()+dist/lam, -log_v, log_u, 1.0)).logsumexp(1).exp()
        ctx.log_v = log_v
        ctx.log_u = log_u
        ctx.dist = dist
        ctx.lam = lam
        return distances

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out[:, None] * ctx.log_u * ctx.lam, grad_out[:, None] * ctx.log_v * ctx.lam, None, None, None

def get_coupling(mu, nu, dist, lam=1e-3, N=1000):
    assert mu.dim() == 2 and nu.dim() == 2 and dist.dim() == 2
    bs = mu.size(0)
    d1, d2 = dist.size()
    assert nu.size(0) == bs and mu.size(1) == d1 and nu.size(1) == d2
    log_mu = mu.log()
    log_nu = nu.log()
    log_u = torch.full_like(mu, -math.log(d1))
    log_v = torch.full_like(nu, -math.log(d2))
    for i in range(N):
        log_v = sinkstep(dist, log_nu, log_u, lam)
        log_u = sinkstep(dist.t(), log_mu, log_v, lam)
    return (log_v[:, None, :]-dist/lam+log_u[:, :, None]).exp()
