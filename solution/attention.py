"""Qwen2 self-attention block on packed (pad-free) tokens: QKV projection with LoRA on q and v,
RoPE, causal attention, o_proj. Forward and backward written out by hand, so autograd keeps only
what the backward really needs.

Per token, the base block keeps (fp16 unless noted): three copies of the projection input
(bitsandbytes saves its input even for frozen weights), the LoRA dropout masks and dropped
inputs, q / k pre-RoPE in fp32 plus the RoPE intermediates, the expanded k / v, the attention
output and o_proj's input. Here:

  * q, k, v come from one tensor-core GEMM over the three NF4 weights unpacked side by side
    (our NF4 kernel), with the fp16 bias in the GEMM epilogue as bitsandbytes' F.linear does;
  * LoRA dropout is a Triton kernel driven by a Philox seed: the mask is never stored, the
    backward regenerates it;
  * both LoRA A matrices run as one [N, 32] GEMM;
  * one Triton kernel adds the LoRA outputs, applies RoPE in fp32 (angles from the position and
    inv_freq, as Qwen2RotaryEmbedding computes them) and scatters q / k / v into the padded
    [B, H, T, 128] layout attention wants; a second one is its exact transpose for the backward;
  * attention is PyTorch's memory-efficient SDPA kernel (is_causal, right padding makes pad keys
    invisible to real queries), recorded in a small local graph;
  * o_proj's weight is frozen: its input is not kept.

Kept per token: the block input (shared with the norm before it), the [N, 32] LoRA A output and
SDPA's own q / k / v / out / logsumexp.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .nf4 import nf4_shape, unpack_nf4


# ----------------------------------------------------------------------------- dropout (Philox)

@triton.jit
def philox_dropout_kernel(x_ptr, out_ptr, n, p, scale, seed, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    keep = tl.rand(seed, offs) >= p
    tl.store(out_ptr + offs, tl.where(keep, x * scale, 0.0).to(out_ptr.dtype.element_ty),
             mask=mask)


@triton.jit
def philox_dropout_bwd_add_kernel(g_ptr, base_ptr, out_ptr, n, p, scale, seed,
                                  BLOCK: tl.constexpr):
    # out = base + dropout_mask(seed) * g * scale   (fp32 math, one rounding)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    keep = tl.rand(seed, offs) >= p
    tl.store(out_ptr + offs, (b + tl.where(keep, g * scale, 0.0)).to(out_ptr.dtype.element_ty),
             mask=mask)


def dropout_fwd(x, p, seed):
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        philox_dropout_kernel[(triton.cdiv(n, 2048),)](x, out, n, p, 1.0 / (1.0 - p), seed,
                                                       BLOCK=2048, num_warps=4)
    return out


def dropout_bwd_add(g, base, p, seed, out_dtype):
    out = torch.empty(base.shape, device=base.device, dtype=out_dtype)
    n = base.numel()
    if n:
        philox_dropout_bwd_add_kernel[(triton.cdiv(n, 2048),)](g, base, out, n, p, 1.0 / (1.0 - p),
                                                               seed, BLOCK=2048, num_warps=4)
    return out


# ----------------------------------------------------------------------------- RoPE + (un)pack

@triton.jit
def rope_pack_kernel(
    qkv_ptr, lq_ptr, lv_ptr,          # [N, QD + 2 KD] fp16, [N, QD] fp16, [N, KD] fp16
    pos_ptr, dst_ptr, inv_freq_ptr,   # [N] int32 position, [N] int64 b*T + t, [HALF] fp32
    q_ptr, k_ptr, v_ptr,              # [B, HQ, T, HD], [B, HK, T, HD] x2, fp16
    T, lora_scale,
    HQ: tl.constexpr, HK: tl.constexpr, HD: tl.constexpr, HAS_LORA: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = HD // 2
    QD: tl.constexpr = HQ * HD
    KD: tl.constexpr = HK * HD
    j = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + tok).to(tl.float32)
    ang = pos * tl.load(inv_freq_ptr + j)
    c = tl.cos(ang)
    s = tl.sin(ang)
    dst = tl.load(dst_ptr + tok)
    b = dst // T
    t = dst % T
    row = qkv_ptr + tok * (QD + 2 * KD)
    for h in tl.static_range(HQ):
        x1 = tl.load(row + h * HD + j).to(tl.float32)
        x2 = tl.load(row + h * HD + HALF + j).to(tl.float32)
        if HAS_LORA:
            x1 += tl.load(lq_ptr + tok * QD + h * HD + j).to(tl.float32) * lora_scale
            x2 += tl.load(lq_ptr + tok * QD + h * HD + HALF + j).to(tl.float32) * lora_scale
        out = q_ptr + ((b * HQ + h) * T + t) * HD
        tl.store(out + j, (x1 * c - x2 * s).to(tl.float16))
        tl.store(out + HALF + j, (x2 * c + x1 * s).to(tl.float16))
    for h in tl.static_range(HK):
        x1 = tl.load(row + QD + h * HD + j).to(tl.float32)
        x2 = tl.load(row + QD + h * HD + HALF + j).to(tl.float32)
        out = k_ptr + ((b * HK + h) * T + t) * HD
        tl.store(out + j, (x1 * c - x2 * s).to(tl.float16))
        tl.store(out + HALF + j, (x2 * c + x1 * s).to(tl.float16))
        v1 = tl.load(row + QD + KD + h * HD + j).to(tl.float32)
        v2 = tl.load(row + QD + KD + h * HD + HALF + j).to(tl.float32)
        if HAS_LORA:
            v1 += tl.load(lv_ptr + tok * KD + h * HD + j).to(tl.float32) * lora_scale
            v2 += tl.load(lv_ptr + tok * KD + h * HD + HALF + j).to(tl.float32) * lora_scale
        out = v_ptr + ((b * HK + h) * T + t) * HD
        tl.store(out + j, v1.to(tl.float16))
        tl.store(out + HALF + j, v2.to(tl.float16))


@triton.jit
def rope_unpack_bwd_kernel(
    dq_ptr, dk_ptr, dv_ptr,           # [B, HQ, T, HD], [B, HQ, T, HD] (per q head), [B, HQ, T, HD]
    pos_ptr, dst_ptr, inv_freq_ptr,
    dqkv_ptr,                         # [N, QD + 2 KD] fp16: gradient of the pre-RoPE q / k / v
    T,
    HQ: tl.constexpr, HK: tl.constexpr, HD: tl.constexpr,
):
    # transpose of rope_pack: rotate back by -angle; k / v gradients summed over the q heads that
    # share each kv head (GQA)
    tok = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = HD // 2
    QD: tl.constexpr = HQ * HD
    KD: tl.constexpr = HK * HD
    GROUP: tl.constexpr = HQ // HK
    j = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + tok).to(tl.float32)
    ang = pos * tl.load(inv_freq_ptr + j)
    c = tl.cos(ang)
    s = tl.sin(ang)
    dst = tl.load(dst_ptr + tok)
    b = dst // T
    t = dst % T
    row = dqkv_ptr + tok * (QD + 2 * KD)
    for h in tl.static_range(HQ):
        src = dq_ptr + ((b * HQ + h) * T + t) * HD
        g1 = tl.load(src + j).to(tl.float32)
        g2 = tl.load(src + HALF + j).to(tl.float32)
        tl.store(row + h * HD + j, (g1 * c + g2 * s).to(tl.float16))
        tl.store(row + h * HD + HALF + j, (g2 * c - g1 * s).to(tl.float16))
    for hk in tl.static_range(HK):
        k1 = tl.zeros([HALF], tl.float32)
        k2 = tl.zeros([HALF], tl.float32)
        w1 = tl.zeros([HALF], tl.float32)
        w2 = tl.zeros([HALF], tl.float32)
        for r in tl.static_range(GROUP):
            h = hk * GROUP + r
            srck = dk_ptr + ((b * HQ + h) * T + t) * HD
            k1 += tl.load(srck + j).to(tl.float32)
            k2 += tl.load(srck + HALF + j).to(tl.float32)
            srcv = dv_ptr + ((b * HQ + h) * T + t) * HD
            w1 += tl.load(srcv + j).to(tl.float32)
            w2 += tl.load(srcv + HALF + j).to(tl.float32)
        tl.store(row + QD + hk * HD + j, (k1 * c + k2 * s).to(tl.float16))
        tl.store(row + QD + hk * HD + HALF + j, (k2 * c - k1 * s).to(tl.float16))
        tl.store(row + QD + KD + hk * HD + j, w1.to(tl.float16))
        tl.store(row + QD + KD + hk * HD + HALF + j, w2.to(tl.float16))


@triton.jit
def gather_rows_kernel(src_ptr, dst_idx_ptr, out_ptr, T, H: tl.constexpr, HD: tl.constexpr):
    # out[tok, h*HD + d] = src[b, h, t, d]  (attention output back to packed [N, H*HD])
    tok = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, HD)
    dst = tl.load(dst_idx_ptr + tok)
    b = dst // T
    t = dst % T
    x = tl.load(src_ptr + ((b * H + h) * T + t) * HD + d)
    tl.store(out_ptr + tok * (H * HD) + h * HD + d, x)


@triton.jit
def scatter_rows_kernel(src_ptr, dst_idx_ptr, out_ptr, T, H: tl.constexpr, HD: tl.constexpr):
    # out[b, h, t, d] = src[tok, h*HD + d]  (its transpose; pads stay zero)
    tok = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, HD)
    dst = tl.load(dst_idx_ptr + tok)
    b = dst // T
    t = dst % T
    x = tl.load(src_ptr + tok * (H * HD) + h * HD + d)
    tl.store(out_ptr + ((b * H + h) * T + t) * HD + d, x)


# ----------------------------------------------------------------------------- the block

class PackInfo:
    """Where each packed token sits in the padded [B, T] batch."""

    def __init__(self, attention_mask):
        mask = attention_mask.bool()
        self.B, self.T = mask.shape
        lengths = mask.sum(1)
        self.right_padded = bool(torch.equal(
            mask, torch.arange(self.T, device=mask.device)[None, :] < lengths[:, None]))
        flat = mask.reshape(-1).nonzero().squeeze(1)
        self.dst = flat                                   # b*T + t for every packed token
        self.pos = (flat % self.T).to(torch.int32)        # right padding: position = t
        self.N = flat.numel()


def dense_qkv(q_w, k_w, v_w):
    shapes = [nf4_shape(w) for w in (q_w, k_w, v_w)]
    rows = sum(s[0] for s in shapes)
    buf = torch.empty(rows, shapes[0][1], device=q_w.device, dtype=torch.float16)
    start = 0
    for w, s in zip((q_w, k_w, v_w), shapes):
        unpack_nf4(w, buf[start:start + s[0]])
        start += s[0]
    return buf


class AttentionBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_q, b_q, a_v, b_v, spec, pack):
        # x [N, D] fp16 packed tokens. a_* [r, D], b_* [out, r] fp32 LoRA weights (trainable).
        # spec: dict with weights / sizes / dropout; pack: PackInfo
        N, D = x.shape
        HQ, HK, HD = spec["hq"], spec["hk"], spec["hd"]
        QD, KD = HQ * HD, HK * HD
        B, T = pack.B, pack.T
        dev = x.device
        p, scale, seed = spec["p"], spec["scaling"], spec["seed"]
        with torch.autocast(device_type=dev.type, enabled=False):
            w_qkv = dense_qkv(spec["q_w"], spec["k_w"], spec["v_w"])
            qkv = torch.addmm(spec["bias16"], x, w_qkv.t()) if spec["bias16"] is not None \
                else torch.mm(x, w_qkv.t())
            del w_qkv
            xd = dropout_fwd(x, p, seed) if p > 0 else x
            a_cat = torch.cat([a_q, a_v]).to(torch.float16)       # [2r, D]
            a = torch.mm(xd, a_cat.t())                           # [N, 2r] fp16
            del xd
            r = a_q.shape[0]
            lq = torch.mm(a[:, :r], b_q.to(torch.float16).t())   # [N, QD]
            lv = torch.mm(a[:, r:], b_v.to(torch.float16).t())   # [N, KD]
            q = torch.zeros(B, HQ, T, HD, device=dev, dtype=torch.float16)
            k = torch.zeros(B, HK, T, HD, device=dev, dtype=torch.float16)
            v = torch.zeros(B, HK, T, HD, device=dev, dtype=torch.float16)
            if N:
                rope_pack_kernel[(N,)](qkv, lq, lv, pack.pos, pack.dst, spec["inv_freq"],
                                       q, k, v, T, scale, HQ=HQ, HK=HK, HD=HD, HAS_LORA=True,
                                       num_warps=2)
            del qkv, lq, lv
            group = HQ // HK
            q.requires_grad_(True)
            k_rep = k.repeat_interleave(group, dim=1).requires_grad_(True)
            v_rep = v.repeat_interleave(group, dim=1).requires_grad_(True)
            del k, v
            with torch.enable_grad():
                o_pad = torch.nn.functional.scaled_dot_product_attention(
                    q, k_rep, v_rep, is_causal=True, scale=spec["softmax_scale"])
            o = torch.empty(N, QD, device=dev, dtype=torch.float16)
            if N:
                gather_rows_kernel[(N, HQ)](o_pad.detach(), pack.dst, o, T, H=HQ, HD=HD,
                                            num_warps=1)
            w_o = unpack_nf4(spec["o_w"])
            out = torch.mm(o, w_o.t())
            del w_o, o
        ctx.spec, ctx.pack = spec, pack
        ctx.graph = (q, k_rep, v_rep, o_pad)
        ctx.save_for_backward(x, a, a_q, b_q, a_v, b_v)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, a, a_q, b_q, a_v, b_v = ctx.saved_tensors
        spec, pack = ctx.spec, ctx.pack
        q, k_rep, v_rep, o_pad = ctx.graph
        ctx.graph = None
        N, D = x.shape
        HQ, HK, HD = spec["hq"], spec["hk"], spec["hd"]
        QD, KD = HQ * HD, HK * HD
        B, T = pack.B, pack.T
        dev = x.device
        p, scale, seed = spec["p"], spec["scaling"], spec["seed"]
        r = a_q.shape[0]
        dout16 = dout.to(torch.float16).contiguous()

        w_o = unpack_nf4(spec["o_w"])
        do = torch.mm(dout16, w_o)                              # [N, QD]
        del w_o
        do_pad = torch.zeros(B, HQ, T, HD, device=dev, dtype=torch.float16)
        if N:
            scatter_rows_kernel[(N, HQ)](do, pack.dst, do_pad, T, H=HQ, HD=HD, num_warps=1)
        del do
        dq, dk, dv = torch.autograd.grad(o_pad, (q, k_rep, v_rep), do_pad)
        del do_pad, o_pad, q, k_rep, v_rep
        dqkv = torch.empty(N, QD + 2 * KD, device=dev, dtype=torch.float16)
        if N:
            rope_unpack_bwd_kernel[(N,)](dq, dk, dv, pack.pos, pack.dst, spec["inv_freq"], dqkv,
                                         T, HQ=HQ, HK=HK, HD=HD, num_warps=2)
        del dq, dk, dv
        dq_pre = dqkv[:, :QD]
        dv_pre = dqkv[:, QD + KD:]
        # LoRA: lq = scale * a_q @ b_q^T  (a_q = a[:, :r]), same for v
        a16 = a
        g_bq = (torch.mm(dq_pre.t(), a16[:, :r]).float() * scale)
        g_bv = (torch.mm(dv_pre.t(), a16[:, r:]).float() * scale)
        da = torch.empty(N, 2 * r, device=dev, dtype=torch.float16)
        torch.mm(dq_pre, b_q.to(torch.float16), out=da[:, :r])
        torch.mm(dv_pre, b_v.to(torch.float16), out=da[:, r:])
        da.mul_(scale)
        xd = dropout_fwd(x, p, seed) if p > 0 else x
        g_a = torch.mm(da.t(), xd).float()                       # [2r, D]
        del xd
        g_aq, g_av = g_a[:r], g_a[r:]
        dx = None
        if ctx.needs_input_grad[0]:
            w_qkv = dense_qkv(spec["q_w"], spec["k_w"], spec["v_w"])
            dx_base = torch.mm(dqkv, w_qkv)                       # [N, D]
            del w_qkv
            a_cat = torch.cat([a_q, a_v]).to(torch.float16)
            dxd = torch.mm(da, a_cat)                              # [N, D]
            if p > 0:
                dx = dropout_bwd_add(dxd, dx_base, p, seed, torch.float16)
            else:
                dx = dx_base + dxd
        return dx, g_aq.to(a_q.dtype), g_bq.to(b_q.dtype), g_av.to(a_v.dtype), \
            g_bv.to(b_v.dtype), None, None
