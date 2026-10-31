"""Qwen2 self-attention block on packed (pad-free) tokens: QKV projection with LoRA on q and v,
RoPE, causal attention, o_proj. Forward and backward written out by hand, so autograd keeps only
what the backward really needs.

Per token, the base block keeps (fp16 unless noted): three copies of the projection input
(bitsandbytes saves its input even for frozen weights), the LoRA dropout masks and dropped
inputs, q / k pre-RoPE in fp32 plus the RoPE intermediates, the expanded k / v, the attention
output and o_proj's input. Here:

  * q, k, v come from one tensor-core GEMM over the three NF4 weights unpacked side by side
    (our NF4 kernel), with the fp16 bias in the GEMM epilogue as bitsandbytes' F.linear does;
  * LoRA dropout uses the very masks the base run draws (same CUDA RNG calls), applied to the
    packed rows in Triton; the backward replays the RNG state instead of storing masks;
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


# ----------------------------------------------------------------------------- LoRA dropout

# The base run applies nn.Dropout(0.05) to the fp32 [B, T, D] norm output, once for q_proj's
# LoRA and once for v_proj's, in that order, in every layer: two draws from the CUDA RNG stream
# per layer. We make the same two draws on a [B, T, D] fp32 tensor of ones, so the masks are the
# very masks the base run uses, and apply them to the packed fp16 rows in Triton. The RNG state
# is recorded, and the backward replays it to rebuild the masks instead of storing them.

@triton.jit
def masked_rows_kernel(x_ptr, mq_ptr, mv_ptr, dst_ptr, xq_ptr, xv_ptr, D,
                       BLOCK: tl.constexpr):
    # xq[n] = x[n] * mq[dst[n]], xv[n] = x[n] * mv[dst[n]]   (masks hold 0 or 1/(1-p))
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    keep = cols < D
    src = tl.load(dst_ptr + tok)
    x = tl.load(x_ptr + tok * D + cols, mask=keep, other=0.0).to(tl.float32)
    mq = tl.load(mq_ptr + src * D + cols, mask=keep, other=0.0)
    mv = tl.load(mv_ptr + src * D + cols, mask=keep, other=0.0)
    tl.store(xq_ptr + tok * D + cols, (x * mq).to(tl.float16), mask=keep)
    tl.store(xv_ptr + tok * D + cols, (x * mv).to(tl.float16), mask=keep)


@triton.jit
def masked_rows_bwd_kernel(base_ptr, gq_ptr, gv_ptr, mq_ptr, mv_ptr, dst_ptr, out_ptr, D,
                           BLOCK: tl.constexpr):
    # out[n] = base[n] + gq[n] * mq[dst[n]] + gv[n] * mv[dst[n]]   (fp32 math, one rounding)
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    keep = cols < D
    src = tl.load(dst_ptr + tok)
    b = tl.load(base_ptr + tok * D + cols, mask=keep, other=0.0).to(tl.float32)
    gq = tl.load(gq_ptr + tok * D + cols, mask=keep, other=0.0).to(tl.float32)
    gv = tl.load(gv_ptr + tok * D + cols, mask=keep, other=0.0).to(tl.float32)
    mq = tl.load(mq_ptr + src * D + cols, mask=keep, other=0.0)
    mv = tl.load(mv_ptr + src * D + cols, mask=keep, other=0.0)
    tl.store(out_ptr + tok * D + cols, (b + gq * mq + gv * mv).to(out_ptr.dtype.element_ty),
             mask=keep)


def _rng_state(device):
    return torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()


def _set_rng_state(state, device):
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


def draw_masks(pack, D, p, device):
    """The two LoRA dropout masks of one layer as the base run draws them: [B*T, D] fp32."""
    ones = torch.ones(pack.B * pack.T, D, device=device, dtype=torch.float32)
    mq = torch.nn.functional.dropout(ones, p, True)
    mv = torch.nn.functional.dropout(ones, p, True)
    return mq, mv


def masked_rows(x, mq, mv, pack):
    N, D = x.shape
    xq, xv = torch.empty_like(x), torch.empty_like(x)
    if N:
        masked_rows_kernel[(N,)](x, mq, mv, pack.dst, xq, xv, D,
                                 BLOCK=triton.next_power_of_2(D), num_warps=4)
    return xq, xv


def masked_rows_bwd(base, gq, gv, mq, mv, pack, out_dtype):
    N, D = base.shape
    out = torch.empty(N, D, device=base.device, dtype=out_dtype)
    if N:
        masked_rows_bwd_kernel[(N,)](base, gq, gv, mq, mv, pack.dst, out, D,
                                     BLOCK=triton.next_power_of_2(D), num_warps=4)
    return out


# ----------------------------------------------------------------------------- RoPE + (un)pack

@triton.jit(do_not_specialize=["T"])
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


@triton.jit(do_not_specialize=["T"])
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


@triton.jit(do_not_specialize=["T"])
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


@triton.jit(do_not_specialize=["T"])
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


def _attention_core(x, a, b_q, b_v, w_qkv, spec, pack):
    """qkv GEMM (+bias), LoRA B, RoPE + scatter to [B, H, T, HD], SDPA. Returns the small autograd
    graph (q, k_rep, v_rep, o_pad) that the backward differentiates."""
    N = x.shape[0]
    HQ, HK, HD = spec["hq"], spec["hk"], spec["hd"]
    B, T = pack.B, pack.T
    dev = x.device
    r = a.shape[1] // 2
    qkv = torch.addmm(spec["bias16"], x, w_qkv.t()) if spec["bias16"] is not None \
        else torch.mm(x, w_qkv.t())
    lq = torch.mm(a[:, :r], b_q.to(torch.float16).t())   # [N, QD]
    lv = torch.mm(a[:, r:], b_v.to(torch.float16).t())   # [N, KD]
    q = torch.zeros(B, HQ, T, HD, device=dev, dtype=torch.float16)
    k = torch.zeros(B, HK, T, HD, device=dev, dtype=torch.float16)
    v = torch.zeros(B, HK, T, HD, device=dev, dtype=torch.float16)
    if N:
        rope_pack_kernel[(N,)](qkv, lq, lv, pack.pos, pack.dst, spec["inv_freq"],
                               q, k, v, T, spec["scaling"], HQ=HQ, HK=HK, HD=HD, HAS_LORA=True,
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
    return q, k_rep, v_rep, o_pad


class AttentionBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_q, b_q, a_v, b_v, spec, pack):
        # x [N, D] fp16 packed tokens. a_* [r, D], b_* [out, r] fp32 LoRA weights (trainable).
        # spec: dict with weights / sizes / dropout / recompute; pack: PackInfo
        N, D = x.shape
        HQ, HD = spec["hq"], spec["hd"]
        QD = HQ * HD
        T = pack.T
        dev = x.device
        p = spec["p"]
        recompute = spec.get("recompute", False)
        with torch.autocast(device_type=dev.type, enabled=False):
            r = a_q.shape[0]
            a = torch.empty(N, 2 * r, device=dev, dtype=torch.float16)
            rng = None
            if p > 0:
                rng = _rng_state(dev)
                mq, mv = draw_masks(pack, D, p, dev)
                xq, xv = masked_rows(x, mq, mv, pack)
                del mq, mv
            else:
                xq = xv = x
            torch.mm(xq, a_q.to(torch.float16).t(), out=a[:, :r])
            torch.mm(xv, a_v.to(torch.float16).t(), out=a[:, r:])
            del xq, xv
            w_qkv = dense_qkv(spec["q_w"], spec["k_w"], spec["v_w"])
            graph = _attention_core(x, a, b_q, b_v, w_qkv, spec, pack)
            del w_qkv
            o_pad = graph[3]
            o = torch.empty(N, QD, device=dev, dtype=torch.float16)
            if N:
                gather_rows_kernel[(N, HQ)](o_pad.detach().contiguous(), pack.dst, o, T, H=HQ,
                                            HD=HD, num_warps=1)
            w_o = unpack_nf4(spec["o_w"])
            out = torch.mm(o, w_o.t())
            del w_o, o
        ctx.spec, ctx.pack, ctx.rng = spec, pack, rng
        ctx.graph = None if recompute else graph
        del graph, o_pad
        ctx.save_for_backward(x, a, a_q, b_q, a_v, b_v)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, a, a_q, b_q, a_v, b_v = ctx.saved_tensors
        spec, pack = ctx.spec, ctx.pack
        N, D = x.shape
        HQ, HK, HD = spec["hq"], spec["hk"], spec["hd"]
        QD, KD = HQ * HD, HK * HD
        B, T = pack.B, pack.T
        dev = x.device
        p, scale = spec["p"], spec["scaling"]
        r = a_q.shape[0]
        dout16 = dout.to(torch.float16).contiguous()

        w_qkv = dense_qkv(spec["q_w"], spec["k_w"], spec["v_w"])
        if ctx.graph is None:      # recompute the attention half of the forward
            q, k_rep, v_rep, o_pad = _attention_core(x, a, b_q, b_v, w_qkv, spec, pack)
        else:
            q, k_rep, v_rep, o_pad = ctx.graph
            ctx.graph = None
        w_o = unpack_nf4(spec["o_w"])
        do = torch.mm(dout16, w_o)                              # [N, QD]
        del w_o
        do_pad = torch.zeros(B, HQ, T, HD, device=dev, dtype=torch.float16)
        if N:
            scatter_rows_kernel[(N, HQ)](do, pack.dst, do_pad, T, H=HQ, HD=HD, num_warps=1)
        del do
        dq, dk, dv = torch.autograd.grad(o_pad, (q, k_rep, v_rep), do_pad)
        del do_pad, o_pad, q, k_rep, v_rep
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        dqkv = torch.empty(N, QD + 2 * KD, device=dev, dtype=torch.float16)
        if N:
            rope_unpack_bwd_kernel[(N,)](dq, dk, dv, pack.pos, pack.dst, spec["inv_freq"], dqkv,
                                         T, HQ=HQ, HK=HK, HD=HD, num_warps=2)
        del dq, dk, dv
        dq_pre = dqkv[:, :QD]
        dv_pre = dqkv[:, QD + KD:]
        # LoRA: lq = scale * a_q @ b_q^T  (a_q = a[:, :r]), same for v
        g_bq = (torch.mm(dq_pre.t(), a[:, :r]).float() * scale)
        g_bv = (torch.mm(dv_pre.t(), a[:, r:]).float() * scale)
        da = torch.empty(N, 2 * r, device=dev, dtype=torch.float16)
        torch.mm(dq_pre, b_q.to(torch.float16), out=da[:, :r])
        torch.mm(dv_pre, b_v.to(torch.float16), out=da[:, r:])
        da.mul_(scale)
        if p > 0:
            current = _rng_state(dev)
            _set_rng_state(ctx.rng, dev)
            mq, mv = draw_masks(pack, D, p, dev)
            _set_rng_state(current, dev)
            xq, xv = masked_rows(x, mq, mv, pack)
        else:
            mq = mv = None
            xq = xv = x
        g_aq = torch.mm(da[:, :r].t(), xq).float()                # [r, D]
        g_av = torch.mm(da[:, r:].t(), xv).float()
        del xq, xv
        dx = None
        if ctx.needs_input_grad[0]:
            dx_base = torch.mm(dqkv, w_qkv)                       # [N, D]
            gq = torch.mm(da[:, :r], a_q.to(torch.float16))       # [N, D]
            gv = torch.mm(da[:, r:], a_v.to(torch.float16))
            if p > 0:
                dx = masked_rows_bwd(dx_base, gq, gv, mq, mv, pack, torch.float16)
            else:
                dx = dx_base + gq + gv
        del w_qkv
        return dx, g_aq.to(a_q.dtype), g_bq.to(b_q.dtype), g_av.to(a_v.dtype), \
            g_bv.to(b_v.dtype), None, None
