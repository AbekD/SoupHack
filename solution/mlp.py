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


@triton.jit
def gated_silu_q8_kernel(gu_ptr, out_ptr, q_ptr, scale_ptr, stride_gu, stride_out, stride_q,
                         stride_scale, I, BLOCK_I: tl.constexpr, SUB: tl.constexpr):
    # out = silu(g) * u (from the exact fp16 gu, as above), and gu kept for the backward as int8
    # with one fp32 scale per SUB values (symmetric, absmax / 127, round to nearest)
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    col = blk * BLOCK_I + tl.arange(0, BLOCK_I)
    keep = col < I
    row_ptr = gu_ptr + row * stride_gu
    g = tl.load(row_ptr + col, mask=keep, other=0.0).to(tl.float32)
    u = tl.load(row_ptr + I + col, mask=keep, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    tl.store(out_ptr + row * stride_out + col, (g * s * u).to(out_ptr.dtype.element_ty), mask=keep)
    NSUB: tl.constexpr = BLOCK_I // SUB
    sub = blk * NSUB + tl.arange(0, NSUB)
    sub_keep = sub * SUB < I
    n_sub = (I + SUB - 1) // SUB
    for half in tl.static_range(2):
        v = tl.reshape(g if half == 0 else u, [NSUB, SUB])
        amax = tl.max(tl.abs(v), axis=1)
        scale = tl.maximum(amax, 1e-30) / 127.0
        r = v / scale[:, None]
        q = tl.where(r >= 0, r + 0.5, r - 0.5).to(tl.int32).to(tl.int8)
        qcol = tl.reshape(col, [NSUB, SUB])
        tl.store(q_ptr + row * stride_q + half * I + qcol, q,
                 mask=tl.reshape(keep, [NSUB, SUB]))
        tl.store(scale_ptr + row * stride_scale + half * n_sub + sub, scale, mask=sub_keep)


@triton.jit
def gated_silu_q8_bwd_kernel(q_ptr, scale_ptr, dout_ptr, dgu_ptr, stride_q, stride_scale,
                             stride_dout, stride_dgu, I, BLOCK_I: tl.constexpr,
                             SUB: tl.constexpr):
    # dgu = [dg | du] from the int8 gu: dg = dout * u * s * (1 + g * (1 - s)), du = dout * g * s
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    keep = col < I
    n_sub = (I + SUB - 1) // SUB
    sub = col // SUB
    sg = tl.load(scale_ptr + row * stride_scale + sub, mask=keep, other=0.0)
    su = tl.load(scale_ptr + row * stride_scale + n_sub + sub, mask=keep, other=0.0)
    g = tl.load(q_ptr + row * stride_q + col, mask=keep, other=0).to(tl.float32) * sg
    u = tl.load(q_ptr + row * stride_q + I + col, mask=keep, other=0).to(tl.float32) * su
    d = tl.load(dout_ptr + row * stride_dout + col, mask=keep, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    silu = g * s
    out = dgu_ptr + row * stride_dgu
    tl.store(out + col, (d * u * (s + silu * (1.0 - s))).to(dgu_ptr.dtype.element_ty), mask=keep)
    tl.store(out + I + col, (d * silu).to(dgu_ptr.dtype.element_ty), mask=keep)


BLOCK_I = 1024
Q8_SUB = 128


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


def gated_silu_q8(gu):
    """(silu(g) * u, int8 gu, fp32 scales)."""
    rows, two_i = gu.shape
    inter = two_i // 2
    out = torch.empty(rows, inter, device=gu.device, dtype=gu.dtype)
    q = torch.empty(rows, two_i, device=gu.device, dtype=torch.int8)
    n_sub = triton.cdiv(inter, Q8_SUB)
    scales = torch.empty(rows, 2 * n_sub, device=gu.device, dtype=torch.float32)
    if rows:
        gated_silu_q8_kernel[(rows, triton.cdiv(inter, BLOCK_I))](
            gu, out, q, scales, gu.stride(0), out.stride(0), q.stride(0), scales.stride(0), inter,
            BLOCK_I=BLOCK_I, SUB=Q8_SUB, num_warps=4)
    return out, q, scales


def gated_silu_q8_backward(q, scales, dout):
    rows, two_i = q.shape
    inter = two_i // 2
    dgu = torch.empty(rows, two_i, device=q.device, dtype=dout.dtype)
    if rows:
        gated_silu_q8_bwd_kernel[(rows, triton.cdiv(inter, BLOCK_I))](
            q, scales, dout, dgu, q.stride(0), scales.stride(0), dout.stride(0), dgu.stride(0),
            inter, BLOCK_I=BLOCK_I, SUB=Q8_SUB, num_warps=4)
    return dgu


def dense_gate_up(gate_w, up_w):
    """[2I, D] fp16: gate rows then up rows."""
    gi = nf4_shape(gate_w)
    ui = nf4_shape(up_w)
    buf = torch.empty(gi[0] + ui[0], gi[1], device=gate_w.device, dtype=torch.float16)
    unpack_nf4(gate_w, buf[:gi[0]])
    unpack_nf4(up_w, buf[gi[0]:])
    return buf


class NF4MLP(torch.autograd.Function):
    """mode: "fp16" keeps gu, "int8" keeps gu as int8 + scales (lossy, gradients only: the
    forward uses the exact fp16 gu), "recompute" keeps nothing but x."""

    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w, mode):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
        x2 = x2.contiguous()
        with torch.autocast(device_type=x.device.type, enabled=False):
            w_gu = dense_gate_up(gate_w, up_w)
            gu = torch.mm(x2, w_gu.t())
            del w_gu
            if mode == "int8":
                act, q, scales = gated_silu_q8(gu)
                del gu
            else:
                act = gated_silu(gu)
            w_d = unpack_nf4(down_w)
            out = torch.mm(act, w_d.t())
            del w_d, act
        ctx.weights = (gate_w, up_w, down_w)
        ctx.mode = mode
        ctx.x_dtype = x.dtype
        if mode == "recompute":
            ctx.save_for_backward(x2)
        elif mode == "int8":
            ctx.save_for_backward(q, scales)
        else:
            ctx.save_for_backward(gu)
        return out.view(*shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        gate_w, up_w, down_w = ctx.weights
        shape = dout.shape
        d2 = dout.reshape(-1, shape[-1]).to(torch.float16).contiguous()
        w_d = unpack_nf4(down_w)
        dact = torch.mm(d2, w_d)        # [N, I]
        del w_d
        w_gu = dense_gate_up(gate_w, up_w)
        if ctx.mode == "int8":
            dgu = gated_silu_q8_backward(saved[0], saved[1], dact)
        else:
            gu = torch.mm(saved[0], w_gu.t()) if ctx.mode == "recompute" else saved[0]
            dgu = gated_silu_backward_(gu, dact)   # in place over gu
        del dact
        dx = torch.mm(dgu, w_gu)        # [N, D]
        return dx.view(*shape[:-1], dx.shape[-1]).to(ctx.x_dtype), None, None, None, None


def nf4_mlp(x, gate_w, up_w, down_w, mode="fp16"):
    if mode is True:
        mode = "recompute"
    elif mode is False:
        mode = "fp16"
    return NF4MLP.apply(x, gate_w, up_w, down_w, mode)
