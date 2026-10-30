"""RMSNorm for Qwen2 (fp32 statistics), forward and backward in Triton, optionally fused with the
residual add that precedes it.

Qwen2RMSNorm in the base run: x.float(); x * rsqrt(mean(x^2) + eps); weight * that, with the
weight in fp32 (prepare_model_for_kbit_training upcasts it). PyTorch runs that as ~6 kernels and
autograd keeps several fp32 [tokens, 1536] intermediates per norm. Here one kernel reads x once,
writes the normalised output (in fp16 when it only feeds fp16 matmuls: the same rounding the
autocast cast would apply) and one fp32 rstd per row; the backward recomputes x_hat from x.

Norm weights are frozen in this task (LoRA only), so the backward returns no weight gradient.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, res_ptr, sum_ptr, w_ptr, out_ptr, rstd_ptr,
    stride_x, stride_res, stride_sum, stride_out, N, eps,
    HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        # residual stream update h = residual + x, written out (it is the next residual)
        r = tl.load(res_ptr + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(sum_ptr + row * stride_sum + cols, x.to(sum_ptr.dtype.element_ty), mask=mask)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dres_ptr, dx_ptr,
    stride_dy, stride_x, stride_dres, stride_dx, N,
    HAS_DRES: tl.constexpr, BLOCK: tl.constexpr,
):
    # dx = rstd * (w*dy - x_hat * mean(w*dy*x_hat)) (+ the residual stream's own gradient)
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    dy = tl.load(dy_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    x_hat = x * rstd
    wdy = w * dy
    c = tl.sum(wdy * x_hat, axis=0) / N
    dx = (wdy - x_hat * c) * rstd
    if HAS_DRES:
        dx += tl.load(dres_ptr + row * stride_dres + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(dx_ptr + row * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _block(n):
    return triton.next_power_of_2(n)


def rmsnorm_fwd(x2d, weight, eps, out_dtype, residual2d=None, sum_dtype=None):
    """y = rmsnorm(x [+ residual]); returns (y, rstd, summed or None). 2-D contiguous rows."""
    rows, n = x2d.shape
    y = torch.empty(rows, n, device=x2d.device, dtype=out_dtype)
    rstd = torch.empty(rows, device=x2d.device, dtype=torch.float32)
    if residual2d is not None:
        summed = torch.empty(rows, n, device=x2d.device, dtype=sum_dtype or residual2d.dtype)
        res, s = residual2d, summed
    else:
        summed, res, s = None, x2d, x2d
    if rows:
        rmsnorm_fwd_kernel[(rows,)](
            x2d, res, s, weight, y, rstd,
            x2d.stride(0), res.stride(0), s.stride(0), y.stride(0), n, eps,
            HAS_RES=residual2d is not None, BLOCK=_block(n), num_warps=4)
    return y, rstd, summed


def rmsnorm_bwd(dy2d, x2d, weight, rstd, dx_dtype, dres2d=None):
    rows, n = x2d.shape
    dx = torch.empty(rows, n, device=x2d.device, dtype=dx_dtype)
    dres = dres2d if dres2d is not None else dy2d
    if rows:
        rmsnorm_bwd_kernel[(rows,)](
            dy2d, x2d, weight, rstd, dres, dx,
            dy2d.stride(0), x2d.stride(0), dres.stride(0), dx.stride(0), n,
            HAS_DRES=dres2d is not None, BLOCK=_block(n), num_warps=4)
    return dx


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype):
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        y, rstd, _ = rmsnorm_fwd(x2d, weight, eps, out_dtype)
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.x_dtype = x.dtype
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, weight, rstd = ctx.saved_tensors
        dy2d = dy.reshape(x2d.shape)
        if not dy2d.is_contiguous():
            dy2d = dy2d.contiguous()
        dx = rmsnorm_bwd(dy2d, x2d, weight, rstd, ctx.x_dtype)
        return dx.view(dy.shape), None, None, None


def rmsnorm(x, weight, eps, out_dtype=None):
    return RMSNormFn.apply(x, weight, eps, out_dtype or x.dtype)
