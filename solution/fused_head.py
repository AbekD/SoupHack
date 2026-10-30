"""Fused output head + causal-LM loss + TRL / Soup metrics, without materialising the
[batch, seq, vocab] logits.

What the base run does at the end of every step (Soup 0.77.0 on a T4):
  logits = lm_head(hidden)              # fp32 weight (tied embedding), fp32 matmul, B*T*V floats
  loss   = ForCausalLMLoss(logits)      # upcast + log_softmax + its gradient: more B*T*V buffers
  TRL: entropy_from_logits(logits), argmax(logits) for mean_token_accuracy
  Soup rewind: per-row cross-entropy from the same logits

Here only the rows that matter go through the head, a chunk of rows at a time:
  * rows with a label (assistant tokens) need the loss, the accuracy and a gradient;
  * the other non-pad rows only need the entropy TRL logs;
  * pad rows need nothing (TRL multiplies their entropy by the attention mask).
For each chunk the head matmul runs on tensor cores (fp16 inputs, fp32 accumulate and fp32
logits when the GPU supports it), then one Triton program per row makes a single streaming
pass over the 151,936 logits (online max / sum-exp / sum-exp*x / argmax, as in the
"online softmax" of Milakov & Gimelshein 2018) and, for labelled rows, a second pass that writes
softmax - onehot (scaled by 2**12 so tiny probabilities survive fp16) over the logits in place.
The gradient on the hidden states is then one more tensor-core matmul with the same weight.

Idea credits: the chunked "fused linear cross-entropy" pattern is from Liger-Kernel
(https://github.com/linkedin/Liger-Kernel) and Apple's Cut Cross-Entropy
(https://arxiv.org/abs/2411.09009); Soup's own `utils/stream_loss_chunk.py` documents what the
trainer metrics need. No code from them is used here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
GRAD_SCALE = 4096.0  # power of two: exact to undo, keeps probabilities ~1e-8 representable
DEFAULT_CHUNK = 512


@triton.jit
def head_row_stats_kernel(
    logits_ptr, logits_row_stride,   # [rows, V] fp32 or fp16
    grad_ptr, grad_row_stride,       # fp16 view over the same storage (only used if WRITE_GRAD)
    target_ptr,                      # [rows] int64, -100 = no label
    lse_ptr, ent_ptr, loss_ptr, hit_ptr,  # [rows] fp32, fp32, fp32, int32
    V, grad_scale,
    WRITE_GRAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * logits_row_stride
    target = tl.load(target_ptr + row)

    cols = tl.arange(0, BLOCK)
    # running statistics, kept as per-lane vectors and merged once at the end
    m_vec = tl.full([BLOCK], float("-inf"), tl.float32)
    s_vec = tl.zeros([BLOCK], tl.float32)
    t_vec = tl.zeros([BLOCK], tl.float32)
    best_v = tl.full([BLOCK], float("-inf"), tl.float32)
    best_i = tl.zeros([BLOCK], tl.int32)
    for start in range(0, V, BLOCK):
        idx = start + cols
        mask = idx < V
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        # per-lane argmax: strictly greater keeps the first (lowest) index, like torch.argmax
        better = x > best_v
        best_i = tl.where(better, idx, best_i)
        best_v = tl.where(better, x, best_v)
        m_new = tl.maximum(m_vec, x)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_vec - m_safe)
        e = tl.exp(x - m_safe)
        xe = tl.where(mask, x * e, 0.0)
        s_vec = s_vec * alpha + e
        t_vec = t_vec * alpha + xe
        m_vec = m_new

    m = tl.max(m_vec, axis=0)
    scale = tl.exp(m_vec - m)
    s = tl.sum(s_vec * scale, axis=0)
    t = tl.sum(t_vec * scale, axis=0)
    lse = m + tl.log(s)
    entropy = lse - t / s
    # argmax: the max value, then the smallest index holding it
    top = tl.max(best_v, axis=0)
    arg = tl.min(tl.where(best_v == top, best_i, 2147483647), axis=0)

    labelled = target != -100
    safe_target = tl.where(labelled, target, 0)
    x_target = tl.load(base + safe_target).to(tl.float32)
    loss = tl.where(labelled, lse - x_target, 0.0)
    hit = (arg.to(tl.int64) == target) & labelled

    tl.store(lse_ptr + row, lse)
    tl.store(ent_ptr + row, entropy)
    tl.store(loss_ptr + row, loss)
    tl.store(hit_ptr + row, hit.to(tl.int32))

    if WRITE_GRAD:
        # softmax - onehot, scaled, written as fp16 over the logits row (the fp16 view of the
        # row sits in the first half of its own fp32 storage, so each block only overwrites
        # values this program has already read)
        gbase = grad_ptr + row * grad_row_stride
        for start in range(0, V, BLOCK):
            idx = start + cols
            mask = idx < V
            x = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(x - lse)
            g = tl.where(idx == target, p - 1.0, p) * grad_scale
            tl.debug_barrier()
            tl.store(gbase + idx, g.to(tl.float16), mask=mask)


def _row_stats(logits, targets, write_grad):
    """Run the row kernel over a [rows, V] logits chunk. Returns (lse, entropy, loss, hit, grad
    view or None)."""
    rows, vocab = logits.shape
    dev = logits.device
    lse = torch.empty(rows, device=dev, dtype=torch.float32)
    ent = torch.empty(rows, device=dev, dtype=torch.float32)
    loss = torch.empty(rows, device=dev, dtype=torch.float32)
    hit = torch.empty(rows, device=dev, dtype=torch.int32)
    if write_grad:
        if logits.dtype == torch.float32:
            # fp16 view: row r starts at fp16 element r * 2 * row_stride
            grad = logits.view(torch.float16)[:, :vocab]
            grad_stride = grad.stride(0)
        else:
            grad, grad_stride = logits, logits.stride(0)
    else:
        grad, grad_stride = logits, 0
    if rows:
        head_row_stats_kernel[(rows,)](
            logits, logits.stride(0), grad, grad_stride, targets,
            lse, ent, loss, hit, vocab, GRAD_SCALE,
            WRITE_GRAD=write_grad, BLOCK=2048, num_warps=8)
    return lse, ent, loss, hit, (grad if write_grad else None)


_MM_FP32_OUT = None  # does torch.mm(fp16, fp16, out_dtype=fp32) run on this GPU?


def _mm(a, b):
    """a @ b with fp16 inputs on tensor cores; fp32 result when supported, else fp16."""
    global _MM_FP32_OUT
    if _MM_FP32_OUT is not False and a.is_cuda:
        try:
            out = torch.mm(a, b, out_dtype=torch.float32)
            _MM_FP32_OUT = True
            return out
        except (RuntimeError, TypeError, NotImplementedError):
            _MM_FP32_OUT = False
    return torch.mm(a, b)


class FusedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight16, targets, active, denominator, chunk):
        # hidden [B, T, D] (any float dtype), weight16 [V, D] fp16, targets [B, T] int64 (already
        # shifted: position t is scored against token t+1), active [B, T] bool (rows needing the
        # entropy), denominator: a python float or 0-d tensor
        B, T, D = hidden.shape
        flat_h = hidden.reshape(-1, D)
        flat_t = targets.reshape(-1)
        labelled = flat_t != IGNORE_INDEX
        need = active.reshape(-1) | labelled
        lab_idx = labelled.nonzero().squeeze(1)
        rest_idx = (need & ~labelled).nonzero().squeeze(1)
        n_lab = lab_idx.numel()

        dev = hidden.device
        n = B * T
        tok_ent = torch.zeros(n, device=dev, dtype=torch.float32)
        tok_loss = torch.zeros(n, device=dev, dtype=torch.float32)
        hits = torch.zeros((), device=dev, dtype=torch.int64)
        w_t = weight16.t()
        with_grad = ctx.needs_input_grad[0]
        dh = torch.empty(n_lab, D, device=dev, dtype=torch.float32) if with_grad else None

        with torch.autocast(device_type=dev.type, enabled=False):
            for idx, grad_rows in ((lab_idx, with_grad), (rest_idx, False)):
                for s in range(0, idx.numel(), chunk):
                    sel = idx[s:s + chunk]
                    h16 = flat_h.index_select(0, sel).to(torch.float16)
                    logits = _mm(h16, w_t)
                    tgt = flat_t.index_select(0, sel)
                    lse, ent, loss, hit, grad = _row_stats(logits, tgt, grad_rows)
                    tok_ent.index_copy_(0, sel, ent)
                    tok_loss.index_copy_(0, sel, loss)
                    hits += hit.sum()
                    if grad_rows:
                        dh[s:s + sel.numel()] = _mm(grad, weight16).float()
                    del logits, grad

        loss_sum = tok_loss.sum()
        loss = loss_sum / denominator
        ctx.save_for_backward(dh, lab_idx)
        ctx.shape, ctx.dtype, ctx.denominator = hidden.shape, hidden.dtype, denominator
        ctx.mark_non_differentiable(tok_ent, tok_loss, hits)
        return loss, tok_ent.view(B, T), tok_loss.view(B, T), hits

    @staticmethod
    def backward(ctx, grad_loss, _g_ent, _g_loss, _g_hits):
        dh, lab_idx = ctx.saved_tensors
        B, T, D = ctx.shape
        grad = torch.zeros(B * T, D, device=grad_loss.device, dtype=torch.float32)
        if dh is not None and dh.numel():
            factor = grad_loss.float() / (ctx.denominator * GRAD_SCALE)
            grad.index_copy_(0, lab_idx, dh * factor)
        return grad.view(B, T, D).to(ctx.dtype), None, None, None, None, None


def shift_labels(labels):
    """Position t is scored against token t + 1; the last position against nothing."""
    return torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def fused_head_loss(hidden, weight16, labels, attention_mask=None, num_items_in_batch=None,
                    chunk=DEFAULT_CHUNK):
    """Loss as ForCausalLMLoss(lm_head(hidden), labels) plus what TRL and Soup log.

    Returns (loss, token_entropy [B, T], token_loss [B, T], correct, labelled)."""
    targets = shift_labels(labels).to(hidden.device)
    if attention_mask is None:
        active = torch.ones(targets.shape, dtype=torch.bool, device=hidden.device)
    else:
        active = attention_mask.to(hidden.device).bool()
    n_labelled = (targets != IGNORE_INDEX).sum()
    if num_items_in_batch is None:
        denominator = n_labelled
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch
    loss, tok_ent, tok_loss, hits = FusedHeadLoss.apply(
        hidden, weight16, targets, active, denominator, chunk)
    return loss, tok_ent, tok_loss, hits, n_labelled
