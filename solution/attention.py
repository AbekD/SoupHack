"""Qwen2 self-attention block on packed (pad-free) tokens: QKV projection with LoRA on q and v,
RoPE, causal attention, o_proj. Forward and backward written out by hand, so only what the
backward really needs is kept, and kept without padding.

  * q, k, v come from one tensor-core GEMM over the three NF4 weights unpacked side by side
    (our NF4 kernel), with the fp16 bias in the GEMM as bitsandbytes' F.linear does;
  * LoRA dropout uses the very masks the base run draws (same CUDA RNG calls), applied to the
    packed rows in Triton; the backward replays the RNG state instead of storing masks;
  * one Triton kernel adds the LoRA outputs and applies RoPE in fp32 (angles from the position
    and inv_freq, as Qwen2RotaryEmbedding computes them), in place over the packed q / k / v;
  * attention is PyTorch's memory-efficient attention kernel (aten op), causal, on the per-row
    padded [B, T, H, 128] layout; right padding makes pad keys invisible to real queries;
  * what the attention backward needs (q, k, v, out) is kept PACKED - real tokens only, k / v
    with 2 kv heads instead of the 12 repeated ones - plus its logsumexp. Rows at pad
    positions never influence a real token's gradient (their output gradient is zero and real
    queries never see pad keys), so the backward rebuilds the padded tensors with zeros there;
  * o_proj's weight is frozen: its input is not kept.

Kept per real token and layer: the block input (shared with the norm before it), the [32] LoRA A
output, the rotated q / k / v (2048 fp16) and the attention output (1536 fp16).
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
    ones = getattr(pack, "ones", None)
    if ones is None or ones.shape != (pack.B * pack.T, D) or ones.device != device:
        ones = pack.ones = torch.ones(pack.B * pack.T, D, device=device, dtype=torch.float32)
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


# ----------------------------------------------------------------------------- RoPE

@triton.jit
def rope_lora_kernel(
    qkv_ptr, lq_ptr, lv_ptr,          # [N, QD + 2 KD] fp16 (in place), [N, QD], [N, KD] fp16
    pos_ptr, inv_freq_ptr,            # [N] int32 positions, [HD // 2] fp32
    lora_scale,
    HQ: tl.constexpr, HK: tl.constexpr, HD: tl.constexpr,
):
    # q += scale * lq, v += scale * lv (fp32, one rounding), then RoPE on q and k, in place
    tok = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = HD // 2
    QD: tl.constexpr = HQ * HD
    KD: tl.constexpr = HK * HD
    j = tl.arange(0, HALF)
    ang = tl.load(pos_ptr + tok).to(tl.float32) * tl.load(inv_freq_ptr + j)
    c = tl.cos(ang)
    s = tl.sin(ang)
    row = qkv_ptr + tok * (QD + 2 * KD)
    for h in tl.static_range(HQ):
        x1 = tl.load(row + h * HD + j).to(tl.float32) + \
            tl.load(lq_ptr + tok * QD + h * HD + j).to(tl.float32) * lora_scale
        x2 = tl.load(row + h * HD + HALF + j).to(tl.float32) + \
            tl.load(lq_ptr + tok * QD + h * HD + HALF + j).to(tl.float32) * lora_scale
        tl.store(row + h * HD + j, (x1 * c - x2 * s).to(tl.float16))
        tl.store(row + h * HD + HALF + j, (x2 * c + x1 * s).to(tl.float16))
    for h in tl.static_range(HK):
        x1 = tl.load(row + QD + h * HD + j).to(tl.float32)
        x2 = tl.load(row + QD + h * HD + HALF + j).to(tl.float32)
        tl.store(row + QD + h * HD + j, (x1 * c - x2 * s).to(tl.float16))
        tl.store(row + QD + h * HD + HALF + j, (x2 * c + x1 * s).to(tl.float16))
        v1 = tl.load(row + QD + KD + h * HD + j).to(tl.float32) + \
            tl.load(lv_ptr + tok * KD + h * HD + j).to(tl.float32) * lora_scale
        v2 = tl.load(row + QD + KD + h * HD + HALF + j).to(tl.float32) + \
            tl.load(lv_ptr + tok * KD + h * HD + HALF + j).to(tl.float32) * lora_scale
        tl.store(row + QD + KD + h * HD + j, v1.to(tl.float16))
        tl.store(row + QD + KD + h * HD + HALF + j, v2.to(tl.float16))


@triton.jit
def rope_inverse_kernel(
    g_ptr,                            # [N, QD + 2 KD] fp16, in place: gradients of rotated q / k
    pos_ptr, inv_freq_ptr,
    HQ: tl.constexpr, HK: tl.constexpr, HD: tl.constexpr,
):
    # transpose of the rotation (rotate by -angle) on the q and k parts; v is left as it is
    tok = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = HD // 2
    QD: tl.constexpr = HQ * HD
    KD: tl.constexpr = HK * HD
    j = tl.arange(0, HALF)
    ang = tl.load(pos_ptr + tok).to(tl.float32) * tl.load(inv_freq_ptr + j)
    c = tl.cos(ang)
    s = tl.sin(ang)
    row = g_ptr + tok * (QD + 2 * KD)
    for h in tl.static_range(HQ + HK):
        g1 = tl.load(row + h * HD + j).to(tl.float32)
        g2 = tl.load(row + h * HD + HALF + j).to(tl.float32)
        tl.store(row + h * HD + j, (g1 * c + g2 * s).to(tl.float16))
        tl.store(row + h * HD + HALF + j, (g2 * c - g1 * s).to(tl.float16))


# ----------------------------------------------------------------------------- the block

class PackInfo:
    """Where each packed token sits in the padded [B, T] batch, plus the head's row sets.

    Built on the CPU from one small device-to-host copy of the attention mask and the labels
    (one sync per step instead of one per nonzero()), and sent back in one copy."""

    def __init__(self, attention_mask, labels=None):
        device = attention_mask.device
        self.B, self.T = attention_mask.shape
        host = attention_mask.detach()
        if labels is not None:
            host = torch.stack([host.to(torch.int64), labels.detach().to(torch.int64)])
        host = host.to("cpu")
        mask = (host[0] if labels is not None else host).bool()
        lengths = mask.sum(1)
        self.right_padded = bool(torch.equal(
            mask, torch.arange(self.T)[None, :] < lengths[:, None]))
        flat = mask.reshape(-1).nonzero().squeeze(1)
        self.N = flat.numel()
        parts = [flat, flat % self.T]
        self.n_lab = self.n_rest = None
        if labels is not None:
            # position t is scored against label t + 1 (the last position against nothing)
            shifted = torch.nn.functional.pad(host[1], (0, 1), value=-100)[:, 1:].reshape(-1)
            targets = shifted.index_select(0, flat)
            labelled = targets != -100
            lab_idx = labelled.nonzero().squeeze(1)
            rest_idx = (~labelled).nonzero().squeeze(1)
            self.n_lab, self.n_rest = lab_idx.numel(), rest_idx.numel()
            parts += [targets, lab_idx, rest_idx]
        packed = torch.cat(parts)
        if device.type == "cuda":
            packed = packed.pin_memory().to(device, non_blocking=True)
        else:
            packed = packed.to(device)
        n = self.N
        self.dst = packed[:n]                               # b*T + t for every packed token
        self.pos = packed[n:2 * n].to(torch.int32)          # right padding: position = t
        if labels is not None:
            self.targets = packed[2 * n:3 * n]              # shifted labels of the packed rows
            start = 3 * n
            self.lab_idx = packed[start:start + self.n_lab]
            self.rest_idx = packed[start + self.n_lab:]


def dense_qkv(q_w, k_w, v_w):
    shapes = [nf4_shape(w) for w in (q_w, k_w, v_w)]
    rows = sum(s[0] for s in shapes)
    buf = torch.empty(rows, shapes[0][1], device=q_w.device, dtype=torch.float16)
    start = 0
    for w, s in zip((q_w, k_w, v_w), shapes):
        unpack_nf4(w, buf[start:start + s[0]])
        start += s[0]
    return buf


def _padded(rows, pack, heads, head_dim, repeat=1):
    """Packed rows [N, heads*hd] -> [B, heads*repeat, T, hd] as a transposed view of a
    [B, T, H, hd] buffer, zeros at pad positions (each kv head repeated `repeat` times)."""
    B, T = pack.B, pack.T
    buf = rows.new_zeros(B * T, heads * head_dim)
    buf.index_copy_(0, pack.dst, rows)
    t = buf.view(B, T, heads, head_dim)
    if repeat > 1:
        t = t.unsqueeze(3).expand(B, T, heads, repeat, head_dim).reshape(B, T, heads * repeat,
                                                                          head_dim)
    return t.transpose(1, 2)


def _packed(t, pack, group=1):
    """[B, H, T, hd] (any layout) -> packed rows [N, H*hd / group], summing `group` adjacent
    heads (the GQA reduction of k / v gradients)."""
    B, H, T, HD = t.shape
    rows = t.transpose(1, 2).reshape(B * T, H, HD)
    if group > 1:
        rows = rows.view(B * T, H // group, group, HD).sum(2, dtype=torch.float32).to(t.dtype)
    return rows.reshape(B * T, -1).index_select(0, pack.dst)


def _eff_attention_fwd(q, k, v, scale):
    """PyTorch's memory-efficient attention (causal), with its logsumexp. Off CUDA (our CPU tests
    under the Triton interpreter) a plain fp32 implementation of the same op."""
    if q.is_cuda:
        return torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, None, True, 0.0, True, scale=scale)
    T = q.shape[2]
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    out = torch.matmul(torch.exp(s - lse[..., None]), v.float()).to(q.dtype)
    return out, lse, torch.zeros(()), torch.zeros(())


def _eff_attention_bwd(do, q, k, v, out, lse, seed, offset, scale):
    if q.is_cuda:
        dq, dk, dv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            do, q, k, v, None, out, lse, seed, offset, 0.0, [True, True, True, False], True,
            scale=scale)
        return dq, dk, dv
    T = q.shape[2]
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    s = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    prob = torch.exp(s - lse[..., None])
    dv = torch.matmul(prob.transpose(-1, -2), dof)
    dp = torch.matmul(dof, vf.transpose(-1, -2))
    ds = prob * (dp - (dof * out.float()).sum(-1, keepdim=True)) * scale
    dq = torch.matmul(ds, kf)
    dk = torch.matmul(ds.transpose(-1, -2), qf)
    return dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype)


def _qkv_rotated(x, a, b_q, b_v, w_qkv, spec, pack):
    """[N, QD + 2 KD] fp16: q / k / v projections + bias + LoRA, RoPE on q and k."""
    N = x.shape[0]
    r = a.shape[1] // 2
    qkv = torch.addmm(spec["bias16"], x, w_qkv.t()) if spec["bias16"] is not None \
        else torch.mm(x, w_qkv.t())
    lq = torch.mm(a[:, :r], b_q.to(torch.float16).t())   # [N, QD]
    lv = torch.mm(a[:, r:], b_v.to(torch.float16).t())   # [N, KD]
    if N:
        rope_lora_kernel[(N,)](qkv, lq, lv, pack.pos, spec["inv_freq"], spec["scaling"],
                               HQ=spec["hq"], HK=spec["hk"], HD=spec["hd"], num_warps=2)
    return qkv


def _attention_fwd(qkv, spec, pack):
    """Causal attention of the packed rotated q / k / v. Returns (packed out [N, QD], lse, seed,
    offset)."""
    HQ, HK, HD = spec["hq"], spec["hk"], spec["hd"]
    QD, KD = HQ * HD, HK * HD
    G = HQ // HK
    q = _padded(qkv[:, :QD], pack, HQ, HD)
    k = _padded(qkv[:, QD:QD + KD], pack, HK, HD, G)
    v = _padded(qkv[:, QD + KD:], pack, HK, HD, G)
    out, lse, seed, offset = _eff_attention_fwd(q, k, v, spec["softmax_scale"])
    del q, k, v
    return _packed(out, pack), lse, seed, offset


class AttentionBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_q, b_q, a_v, b_v, spec, pack):
        # x [N, D] fp16 packed tokens. a_* [r, D], b_* [out, r] fp32 LoRA weights (trainable).
        # spec: dict with weights / sizes / dropout / recompute; pack: PackInfo
        N, D = x.shape
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
            qkv = _qkv_rotated(x, a, b_q, b_v, w_qkv, spec, pack)
            del w_qkv
            o, lse, seed, offset = _attention_fwd(qkv, spec, pack)
            w_o = unpack_nf4(spec["o_w"])
            out = torch.mm(o, w_o.t())
            del w_o
        ctx.spec, ctx.pack, ctx.rng, ctx.recompute = spec, pack, rng, recompute
        if recompute:
            del qkv, o, lse, seed, offset
            ctx.save_for_backward(x, a, a_q, b_q, a_v, b_v)
        else:
            ctx.save_for_backward(x, a, a_q, b_q, a_v, b_v, qkv, o, lse, seed, offset)
        return out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        x, a, a_q, b_q, a_v, b_v = saved[:6]
        spec, pack = ctx.spec, ctx.pack
        N, D = x.shape
        HQ, HK, HD = spec["hq"], spec["hk"], spec["hd"]
        QD, KD = HQ * HD, HK * HD
        G = HQ // HK
        dev = x.device
        p, scale = spec["p"], spec["scaling"]
        r = a_q.shape[0]
        dout16 = dout.to(torch.float16).contiguous()

        w_qkv = dense_qkv(spec["q_w"], spec["k_w"], spec["v_w"])
        if ctx.recompute:          # recompute the attention half of the forward
            qkv = _qkv_rotated(x, a, b_q, b_v, w_qkv, spec, pack)
            o, lse, seed, offset = _attention_fwd(qkv, spec, pack)
        else:
            qkv, o, lse, seed, offset = saved[6:]
        w_o = unpack_nf4(spec["o_w"])
        do = torch.mm(dout16, w_o)                              # [N, QD]
        del w_o
        q = _padded(qkv[:, :QD], pack, HQ, HD)
        k = _padded(qkv[:, QD:QD + KD], pack, HK, HD, G)
        v = _padded(qkv[:, QD + KD:], pack, HK, HD, G)
        out_p = _padded(o, pack, HQ, HD)
        do_p = _padded(do, pack, HQ, HD)
        del do, o
        dq, dk, dv = _eff_attention_bwd(do_p, q, k, v, out_p, lse, seed, offset,
                                        spec["softmax_scale"])
        del q, k, v, out_p, do_p
        dqkv = torch.cat([_packed(dq, pack), _packed(dk, pack, G), _packed(dv, pack, G)], dim=1)
        del dq, dk, dv
        if N:
            rope_inverse_kernel[(N,)](dqkv, pack.pos, spec["inv_freq"], HQ=HQ, HK=HK, HD=HD,
                                      num_warps=2)
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
