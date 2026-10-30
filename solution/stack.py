"""Our forward of the whole Qwen2 decoder stack for the training step, on packed (pad-free)
tokens.

Right padding means every real token sees exactly the tokens it sees in the padded batch, at
the same positions, so dropping the pads changes no value; it removes ~40% of the rows every
matmul and element-wise op would otherwise process. Rows of a batch never mix: attention runs on
the padded [B, H, T, D] layout (causal, so pad keys are invisible to real queries).

Residual stream: fp32, as in the base run (the embedding is read in fp32 and fp16 block outputs
are added to it). Norm outputs are fp16 (they only feed fp16 matmuls).
"""

from __future__ import annotations

import torch

from .attention import AttentionBlock, PackInfo
from .mlp import NF4MLP
from .nf4 import is_nf4
from .rmsnorm import rmsnorm_bwd, rmsnorm_fwd

_GEN = None


def _seed():
    """Dropout seeds from a private generator: no draw from the global CPU / CUDA RNG streams."""
    global _GEN
    if _GEN is None:
        _GEN = torch.Generator()
        _GEN.manual_seed(torch.initial_seed() ^ 0x5EED)
    return int(torch.randint(0, 2**31 - 1, (1,), generator=_GEN).item())


class AddRMSNorm(torch.autograd.Function):
    """h_out = h + delta (fp32 residual stream), y = rmsnorm(h_out) in fp16.

    Saves only y (fp16, usually kept by the next block anyway) and one rstd per row: the backward
    recovers x_hat = y / w (exact up to the fp16 rounding of y; norm weights have no zeros, which
    is checked), so the fp32 residual is never kept."""

    @staticmethod
    def forward(ctx, h, delta, weight, eps):
        if delta is None:
            y, rstd, _ = rmsnorm_fwd(h, weight, eps, torch.float16)
            h_out = h
        else:
            y, rstd, h_out = rmsnorm_fwd(delta, weight, eps, torch.float16, residual2d=h,
                                         sum_dtype=torch.float32)
        ctx.save_for_backward(y, weight, rstd)
        ctx.has_delta = delta is not None
        ctx.delta_dtype = delta.dtype if delta is not None else None
        return h_out, y

    @staticmethod
    def backward(ctx, dh_out, dy):
        y, weight, rstd = ctx.saved_tensors
        x_hat = y.float() / weight          # [N, D] fp32 temporary
        # rmsnorm_bwd with x = x_hat and rstd = 1 gives (w*dy - x_hat*c), then * rstd per row
        if dy is None:
            dh = dh_out
        else:
            ones = torch.ones_like(rstd)
            dxh = rmsnorm_bwd(dy.contiguous(), x_hat, weight, ones, torch.float32)
            dh = dxh * rstd[:, None]
            if dh_out is not None:
                dh = dh + dh_out
        del x_hat
        ddelta = dh.to(ctx.delta_dtype) if ctx.has_delta else None
        return dh, ddelta, None, None


def _lora_parts(proj):
    """(bnb Linear4bit, A, B, dropout p, scaling) for a peft LoRA Linear4bit, or None."""
    adapters = getattr(proj, "active_adapters", None)
    if not adapters or len(adapters) != 1 or getattr(proj, "merged", False):
        return None
    name = adapters[0]
    if getattr(proj, "use_dora", {}).get(name, False):
        return None
    base = proj.base_layer
    a = proj.lora_A[name].weight
    b = proj.lora_B[name].weight
    drop = proj.lora_dropout[name]
    p = float(getattr(drop, "p", 0.0)) if isinstance(drop, torch.nn.Dropout) else 0.0
    if not isinstance(drop, (torch.nn.Dropout, torch.nn.Identity)):
        return None
    return base, a, b, p, float(proj.scaling[name])


def stack_supported(qwen2_model):
    """Is this the configuration our hand-written stack covers?"""
    cfg = qwen2_model.config
    if getattr(cfg, "hidden_act", "silu") != "silu" or getattr(cfg, "use_sliding_window", False):
        return False
    if getattr(cfg, "attention_dropout", 0.0):
        return False
    rope = getattr(qwen2_model, "rotary_emb", None)
    if rope is None or getattr(rope, "attention_scaling", 1.0) != 1.0 \
            or getattr(rope, "rope_type", "default") != "default":
        return False
    for layer in qwen2_model.layers:
        att, mlp = layer.self_attn, layer.mlp
        q, v = _lora_parts(att.q_proj), _lora_parts(att.v_proj)
        if q is None or v is None or q[3] != v[3] or q[4] != v[4] or q[1].shape[0] != v[1].shape[0]:
            return False
        for base in (q[0], att.k_proj, v[0], att.o_proj, mlp.gate_proj, mlp.up_proj,
                     mlp.down_proj):
            if type(base).__name__ != "Linear4bit" or not is_nf4(base.weight):
                return False
        if att.o_proj.bias is not None or any(m.bias is not None for m in
                                              (mlp.gate_proj, mlp.up_proj, mlp.down_proj)):
            return False
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if norm.weight.requires_grad or bool((norm.weight == 0).any()):
                return False
    norm = qwen2_model.norm
    return not norm.weight.requires_grad and not bool((norm.weight == 0).any())


def _layer_spec(layer, cfg, inv_freq):
    att = layer.self_attn
    qb, aq, bq, p, scaling = _lora_parts(att.q_proj)
    vb, av, bv, _, _ = _lora_parts(att.v_proj)
    biases = [m.bias for m in (qb, att.k_proj, vb)]
    bias16 = torch.cat([b.detach().to(torch.float16) for b in biases]) \
        if all(b is not None for b in biases) else None
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    spec = dict(q_w=qb.weight, k_w=att.k_proj.weight, v_w=vb.weight, o_w=att.o_proj.weight,
                bias16=bias16, hq=cfg.num_attention_heads, hk=cfg.num_key_value_heads,
                hd=head_dim, p=p, scaling=scaling, inv_freq=inv_freq,
                softmax_scale=head_dim ** -0.5)
    return spec, (aq, bq, av, bv)


def stack_forward(qwen2_model, input_ids, attention_mask, mlp_recompute=False):
    """Final-norm hidden states [N, D] fp16 of the real tokens, and the PackInfo."""
    cfg = qwen2_model.config
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pack = PackInfo(attention_mask)
    if not pack.right_padded:
        return None, None
    ids = input_ids.reshape(-1).index_select(0, pack.dst)
    h = qwen2_model.embed_tokens(ids)                       # [N, D] fp32 (our fp16->fp32 lookup)
    if h.dtype != torch.float32:
        h = h.float()
    inv_freq = qwen2_model.rotary_emb.inv_freq.float().contiguous()
    delta = None
    eps = cfg.rms_norm_eps
    for layer in qwen2_model.layers:
        spec, (aq, bq, av, bv) = _layer_spec(layer, cfg, inv_freq)
        spec["seed"] = _seed() if spec["p"] > 0 and qwen2_model.training else 0
        if not qwen2_model.training:
            spec["p"] = 0.0
        h, y = AddRMSNorm.apply(h, delta, layer.input_layernorm.weight, eps)
        attn = AttentionBlock.apply(y, aq, bq, av, bv, spec, pack)
        h, y = AddRMSNorm.apply(h, attn, layer.post_attention_layernorm.weight, eps)
        mlp = layer.mlp
        delta = NF4MLP.apply(y, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight,
                             mlp_recompute)
    h, y = AddRMSNorm.apply(h, delta, qwen2_model.norm.weight, eps)
    return y, pack
