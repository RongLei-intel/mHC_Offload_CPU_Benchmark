from __future__ import annotations

import torch
from torch import nn

try:
    import sgl_kernel
except ImportError:
    pass

class MHC(nn.Module):
    """Manifold-Constrained Hyper-Connections (mHC) module."""

    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        residual: nn.Module | None = None,
        tmax: int = 20,
        rms_eps: float = 1e-5,
        alpha_init: float = 0.01,
        data_type: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")

        self.dim = dim
        self.n_streams = n_streams
        self.tmax = tmax
        self.residual = residual if residual is not None else nn.Identity()

        self.norm = nn.RMSNorm(n_streams * dim, eps=rms_eps).to(dtype=data_type)

        self.phi_pre = nn.Linear(n_streams * dim, n_streams, bias=False, dtype=data_type)
        self.phi_post = nn.Linear(n_streams * dim, n_streams, bias=False, dtype=data_type)
        self.phi_res = nn.Linear(n_streams * dim, n_streams * n_streams, bias=False, dtype=data_type)

        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init, dtype=data_type))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init, dtype=data_type))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init, dtype=data_type))

        self.b_pre = nn.Parameter(torch.zeros(n_streams, dtype=data_type))
        self.b_post = nn.Parameter(torch.zeros(n_streams, dtype=data_type))
        self.b_res = nn.Parameter(torch.zeros(n_streams * n_streams, dtype=data_type))

    def init_streams(self, x: torch.Tensor) -> torch.Tensor:
        """Lift [B, T, C] input to [B, T, n, C] streams by replication."""
        if x.dim() != 3:
            raise ValueError("x must have shape [B, T, C]")
        return x.unsqueeze(-2).expand(-1, -1, self.n_streams, -1).contiguous()

    def collapse_streams(self, x_streams: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Collapse [B, T, n, C] streams to [B, T, C]."""
        if x_streams.dim() != 4:
            raise ValueError("x_streams must have shape [B, T, n, C]")
        if mode == "mean":
            return x_streams.mean(dim=-2)
        if mode == "sum":
            return x_streams.sum(dim=-2)
        raise ValueError(f"Unknown collapse mode: {mode}")
    
    def cal_H_res(self, x_streams: torch.Tensor, SGL_AVAILABLE=False) ->  torch.Tensor:
        bsz, seq_len, n_streams, dim = x_streams.shape
        if SGL_AVAILABLE:
            x_flat = x_streams.view(-1, n_streams * dim)
        else:
            x_flat = x_streams.view(bsz, seq_len, n_streams * dim)
        x_norm = self.norm(x_flat)
        return self.alpha_res * self.phi_res(x_norm) + self.b_res

    def cal_sinkhorn(self, h_res_logits):
        shape = h_res_logits.shape[:-1] + (self.n_streams, self.n_streams)
        h_res = h_res_logits.view(shape)
        return sinkhorn_knopp(h_res, tmax=self.tmax, return_dtype=h_res.dtype)

    def cal_app(self, h_res_matrix, x_streams, SGL_AVAILABLE=False, out=None):
        if SGL_AVAILABLE:
            return torch.ops.sgl_kernel.bmm_cpu(out, h_res_matrix.view(-1, self.n_streams, self.n_streams), x_streams, True, None)
        else:
            return torch.einsum("btij,btjc->btic", h_res_matrix, x_streams)


    def cal_total(self, x_streams: torch.Tensor, app_x_streams: torch.Tensor, SGL_AVAILABLE=False, out=None) -> torch.Tensor:
        h_res_logits = self.cal_H_res(x_streams, SGL_AVAILABLE)
        h_res_matrix = self.cal_sinkhorn(h_res_logits)
        return self.cal_app(h_res_matrix, app_x_streams, SGL_AVAILABLE, out=out)

   
    def _compute_hyper_connections(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, n, dim = x_streams.shape
        if n != self.n_streams or dim != self.dim:
            raise ValueError("x_streams must have shape [B, T, n_streams, dim]")

        x_flat = x_streams.reshape(bsz, seq_len, n * dim)
        x_norm = self.norm(x_flat)

        h_pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre
        h_post = self.alpha_post * self.phi_post(x_norm) + self.b_post
        h_res = self.alpha_res * self.phi_res(x_norm) + self.b_res

        h_pre = torch.sigmoid(h_pre)
        h_post = 2.0 * torch.sigmoid(h_post)
        h_res = h_res.view(bsz, seq_len, n, n)
        h_res = sinkhorn_knopp(h_res, tmax=self.tmax, return_dtype=h_res.dtype)

        return h_pre, h_post, h_res

    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        h_pre, h_post, h_res = self._compute_hyper_connections(x_streams)

        x_in = torch.einsum("btn,btnc->btc", h_pre, x_streams)
        y = self.residual(x_in)

        resmix = torch.einsum("btij,btjc->btic", h_res, x_streams)
        addy = h_post.unsqueeze(-1) * y.unsqueeze(-2)

        return resmix + addy


class SglRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size,dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return torch.ops.sgl_kernel.rmsnorm_cpu(x, self.weight, self.eps)
    
class SglLinear(nn.Module):
    def __init__(self, K, N, bias=True,dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, K, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.randn(N,dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.packed_weight = torch.ops.sgl_kernel.convert_weight_packed(self.weight)

    def forward(self, input):
        return torch.ops.sgl_kernel.weight_packed_linear(
            input, self.packed_weight, self.bias, True
        )
    

class MHC_Sglang(MHC):
    """mHC module optimized with sglang kernel"""

    def __init__(self, dim, n_streams = 4, residual = None, tmax = 20, rms_eps = 0.00001, alpha_init = 0.01, data_type = torch.bfloat16):
        super().__init__(dim, n_streams, residual, tmax, rms_eps, alpha_init, data_type)

        self.norm = SglRMSNorm(n_streams * dim, eps=rms_eps, dtype=data_type)

        self.phi_pre = SglLinear(n_streams * dim, n_streams, bias=False, dtype=data_type)
        self.phi_post = SglLinear(n_streams * dim, n_streams, bias=False, dtype=data_type)
        self.phi_res = SglLinear(n_streams * dim, n_streams * n_streams, bias=False, dtype=data_type)


@torch.no_grad()
def _sinkhorn_knopp(
    logits: torch.Tensor,
    tmax: int = 20,
    eps: float = 1e-8,
    return_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sinkhorn-Knopp normalization for doubly-stochastic matrices.

    Args:
        logits: Tensor with shape [..., n, n].
        tmax: number of normalization iterations.
        eps: stability epsilon to avoid division by zero.
        return_dtype: optional dtype to cast output to.

    Returns:
        Doubly-stochastic matrices with same shape as logits.
    """

    orig_dtype = logits.dtype
    x = logits.float()
    x = x - x.amax(dim=(-2, -1), keepdim=True)
    x = torch.exp(x)

    for _ in range(tmax):
        col_sum = x.sum(dim=-2, keepdim=True)
        x = x / (col_sum + eps)
        row_sum = x.sum(dim=-1, keepdim=True)
        x = x / (row_sum + eps)

    if return_dtype is None:
        return_dtype = orig_dtype
    return x.to(return_dtype)

try: 
    sinkhorn_knopp = torch.compile(_sinkhorn_knopp,fullgraph=True)
except Exception:
    print("torch.compile failed for sinkhorn_knopp, using uncompiled version.")
