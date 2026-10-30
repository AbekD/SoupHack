"""Qwen2 MLP, down(silu(gate(x)) * up(x)), on frozen NF4 weights, with what autograd keeps cut
to the bone.

Why the base MLP is heavy: per token and layer autograd keeps the fp16 input of every 4-bit
matmul (bitsandbytes saves it even though the weight is frozen), gate, up, silu(gate) and the
product for down_proj: about 75 KB per token per layer, 28 layers deep.

Here:
  * gate and up are unpacked (our NF4 kernel) into one [2I, D] fp16 matrix and run as a single
    tensor-core GEMM giving gu = [N, 2I];
  * one Triton kernel turns gu into silu(g) * u, reading both halves of each row;
  * down_proj is another GEMM. Its weight is frozen, so its input is never saved: the backward
    only needs grad @ W;
  * the backward kernel rebuilds silu from gu and writes d(gu) over gu in place, then
    dx = d(gu) @ W_gu.
  * With `recompute=True` only x (fp16, D values per token) is kept and gu is recomputed in the
    backward with one extra GEMM; otherwise gu (2I values per token) is kept.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .nf4 import nf4_shape, unpack_nf4


@triton.jit
def gated_silu_rows_kernel(gu_ptr, out_ptr, stride_gu, stride_out, I,
                           BLOCK_I: tl.constexpr):
    # one program per (row, column tile): out[r, c] = silu(gu[r, c]) * gu[r, I + c]
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    keep = col < I
    row_ptr = gu_ptr + row * stride_gu
    g = tl.load(row_ptr + col, mask=keep, other=0.0).to(tl.float32)
    u = tl.load(row_ptr + I + col, mask=keep, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    tl.store(out_ptr + row * stride_out + col, (g * s * u).to(out_ptr.dtype.element_ty), mask=keep)


@triton.jit
def gated_silu_rows_bwd_kernel(gu_ptr, dout_ptr, stride_gu, stride_dout, I,
                               BLOCK_I: tl.constexpr):
    # in place over gu: [dg | du] with dg = dout * u * s * (1 + g * (1 - s)), du = dout * g * s
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    keep = col < I
    row_ptr = gu_ptr + row * stride_gu
    g = tl.load(row_ptr + col, mask=keep, other=0.0).to(tl.float32)
    u = tl.load(row_ptr + I + col, mask=keep, other=0.0).to(tl.float32)
    d = tl.load(dout_ptr + row * stride_dout + col, mask=keep, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    silu = g * s
    du = d * silu
    dg = d * u * (s + silu * (1.0 - s))
    tl.store(row_ptr + col, dg.to(gu_ptr.dtype.element_ty), mask=keep)
    tl.store(row_ptr + I + col, du.to(gu_ptr.dtype.element_ty), mask=keep)


BLOCK_I = 1024


def gated_silu(gu):
    rows, two_i = gu.shape
    inter = two_i // 2
    out = torch.empty(rows, inter, device=gu.device, dtype=gu.dtype)
    if rows:
        gated_silu_rows_kernel[(rows, triton.cdiv(inter, BLOCK_I))](
            gu, out, gu.stride(0), out.stride(0), inter, BLOCK_I=BLOCK_I, num_warps=4)
    return out


def gated_silu_backward_(gu, dout):
    rows, two_i = gu.shape
    inter = two_i // 2
    if rows:
        gated_silu_rows_bwd_kernel[(rows, triton.cdiv(inter, BLOCK_I))](
            gu, dout, gu.stride(0), dout.stride(0), inter, BLOCK_I=BLOCK_I, num_warps=4)
    return gu


def dense_gate_up(gate_w, up_w):
    """[2I, D] fp16: gate rows then up rows."""
    gi = nf4_shape(gate_w)
    ui = nf4_shape(up_w)
    buf = torch.empty(gi[0] + ui[0], gi[1], device=gate_w.device, dtype=torch.float16)
    unpack_nf4(gate_w, buf[:gi[0]])
    unpack_nf4(up_w, buf[gi[0]:])
    return buf


class NF4MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w, recompute):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
        x2 = x2.contiguous()
        with torch.autocast(device_type=x.device.type, enabled=False):
            w_gu = dense_gate_up(gate_w, up_w)
            gu = torch.mm(x2, w_gu.t())
            del w_gu
            act = gated_silu(gu)
            w_d = unpack_nf4(down_w)
            out = torch.mm(act, w_d.t())
            del w_d, act
        ctx.weights = (gate_w, up_w, down_w)
        ctx.recompute = recompute
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(x2 if recompute else gu)
        return out.view(*shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        (saved,) = ctx.saved_tensors
        gate_w, up_w, down_w = ctx.weights
        shape = dout.shape
        d2 = dout.reshape(-1, shape[-1]).to(torch.float16).contiguous()
        w_gu = dense_gate_up(gate_w, up_w)
        if ctx.recompute:
            gu = torch.mm(saved, w_gu.t())
        else:
            gu = saved
        w_d = unpack_nf4(down_w)
        dact = torch.mm(d2, w_d)        # [N, I]
        del w_d
        dgu = gated_silu_backward_(gu, dact)   # in place over gu
        del dact
        dx = torch.mm(dgu, w_gu)        # [N, D]
        return dx.view(*shape[:-1], dx.shape[-1]).to(ctx.x_dtype), None, None, None, None


def nf4_mlp(x, gate_w, up_w, down_w, recompute=False):
    return NF4MLP.apply(x, gate_w, up_w, down_w, recompute)
