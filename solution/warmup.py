"""Compile our Triton kernels while Soup loads the model.

Every measured run starts with empty Triton caches, and compiling our kernels lazily at their
first launch stalled the first training step by ~7 s on the T4 (step 1: 7.9 s vs 1.1 s for the
base). Compilation is CPU work and mostly runs outside the GIL, and loading + quantising the
checkpoint takes ~15 s, so `apply()` starts one thread that compiles every kernel variant the
training step will launch (Triton's own `JITFunction.warmup`: compile only, nothing runs on the
GPU), with the argument types and the integer values that decide Triton's specialisation, and
also builds each kernel's C launcher (Triton compiles it with gcc at first launch, ~0.2-0.5 s
each on Kaggle) and loads its binary. The first step then finds them all ready. The compile time still counts in the
measured wall time; it just overlaps with model loading. If a variant was not warmed (another
model shape), it compiles at first launch as before.

The shapes are Qwen2.5-1.5B's (hidden 1536, MLP 8960, vocab 151936, 12 query / 2 kv heads of 128),
the model of this task.
"""

from __future__ import annotations

import threading

import torch

_THREAD = None


def _variants(mlp_mode="int8"):
    """Only the variants this run's settings launch (MLP: int8 or fp16 activations)."""
    from .attention import (masked_rows_bwd_kernel, masked_rows_kernel, rope_inverse_kernel,
                            rope_lora_kernel)
    from .fused_head import GRAD_SCALE, head_row_stats_kernel
    from .mlp import (BLOCK_I, Q8_SUB, gated_silu_q8_bwd_kernel, gated_silu_q8_kernel,
                      gated_silu_rows_bwd_kernel, gated_silu_rows_kernel)
    from .nf4 import _EXACT, nf4_unpack_kernel
    from .rmsnorm import add_rmsnorm_bwd_kernel, rmsnorm_fwd_kernel

    f16, f32, i32, i64, u8 = torch.float16, torch.float32, torch.int32, torch.int64, torch.uint8
    i8 = torch.int8
    D, I, V, HQ, HK, HD = 1536, 8960, 151936, 12, 2, 128
    BD = 2048  # next power of two of D
    one = (1,)
    return [
        # NF4 unpack (double quant, blocks of 64, nested blocks of 256)
        (nf4_unpack_kernel, (i32, u8, f32, f32, f32, i32, 0.5, 1024),
         dict(BLOCKSIZE=64, NESTED_BLOCK=256, NESTED=True, EXACT=_EXACT, WORDS=512, num_warps=4)),
        # residual + RMSNorm: first layer (no residual, fp32 input) and every other one
        (rmsnorm_fwd_kernel, (f32, f32, f32, f32, f16, f32, D, D, D, D, D, 1e-6),
         dict(HAS_RES=False, BLOCK=BD, num_warps=4)),
        (rmsnorm_fwd_kernel, (f16, f32, f32, f32, f16, f32, D, D, D, D, D, 1e-6),
         dict(HAS_RES=True, BLOCK=BD, num_warps=4)),
        (add_rmsnorm_bwd_kernel, (f16, f16, f32, f32, f32, f32, f16, D),
         dict(HAS_DRES=True, WRITE_DD=True, BLOCK=BD, num_warps=4)),
        (add_rmsnorm_bwd_kernel, (f16, f16, f32, f32, f32, f32, f16, D),
         dict(HAS_DRES=False, WRITE_DD=True, BLOCK=BD, num_warps=4)),
        # attention
        (masked_rows_kernel, (f16, f32, f32, i64, f16, f16, D), dict(BLOCK=BD, num_warps=4)),
        (masked_rows_bwd_kernel, (f16, f16, f16, f32, f32, i64, f16, D),
         dict(BLOCK=BD, num_warps=4)),
        (rope_lora_kernel, (f16, f16, f16, i32, f32, 2.0), dict(HQ=HQ, HK=HK, HD=HD, num_warps=2)),
        (rope_inverse_kernel, (f16, i32, f32), dict(HQ=HQ, HK=HK, HD=HD, num_warps=2)),
        # MLP
    ] + ([
        (gated_silu_q8_kernel, (f16, f16, i8, f32, 2 * I, I, 2 * I, 2 * (I // Q8_SUB), I),
         dict(BLOCK_I=BLOCK_I, SUB=Q8_SUB, num_warps=4)),
        (gated_silu_q8_bwd_kernel, (i8, f32, f16, f16, 2 * I, 2 * (I // Q8_SUB), I, 2 * I, I),
         dict(BLOCK_I=BLOCK_I, SUB=Q8_SUB, num_warps=4)),
    ] if mlp_mode == "int8" else [
        (gated_silu_rows_kernel, (f16, f16, 2 * I, I, I), dict(BLOCK_I=BLOCK_I, num_warps=4)),
        (gated_silu_rows_bwd_kernel, (f16, f16, 2 * I, I, I), dict(BLOCK_I=BLOCK_I, num_warps=4)),
    ]) + [
        # output head: labelled rows (with gradient) and the other rows (entropy only)
        (head_row_stats_kernel, (f32, V, f16, 2 * V, i64, f32, f32, f32, i32, V, GRAD_SCALE),
         dict(WRITE_GRAD=True, BLOCK=2048, num_warps=8)),
        (head_row_stats_kernel, (f32, V, f32, 0, i64, f32, f32, f32, i32, V, GRAD_SCALE),
         dict(WRITE_GRAD=False, BLOCK=2048, num_warps=8)),
    ], one


import os as _os

# Kaggle gives 4 CPU cores; compilation is mostly outside the GIL (dev knob for A/B runs)
THREADS = max(1, int(_os.environ.get("SOUP_KERNELS_WARMUP_THREADS", "2")))


def _compile(variants, grid):
    try:
        torch.cuda.set_device(0)
    except Exception:  # noqa: BLE001
        return
    for kernel, args, kwargs in variants:
        try:
            compiled = kernel.warmup(*args, grid=grid, **kwargs)
            if hasattr(compiled, "result"):
                compiled = compiled.result()
            # Triton builds each kernel's C launcher (a gcc call) and loads its binary only at the
            # first launch; do that here too, so the first training step does neither
            init = getattr(compiled, "_init_handles", None)
            if callable(init):
                init()
        except Exception:  # noqa: BLE001 - warming up is an optimisation, never a failure
            pass


def _warm_cuda():
    """Load the cuBLAS / attention kernels the step uses (CUDA loads kernels lazily, at first
    launch) and create the cuBLAS handle, by running the step's matmul and attention shapes once
    on small inputs. Nothing here draws random numbers, so the CUDA RNG stream (the LoRA dropout
    masks) is untouched; the buffers are freed when this returns."""
    D, QKV, I, V, HQ, HD = 1536, 2048, 8960, 151936, 12, 128
    dev = torch.device("cuda", 0)
    f16 = torch.float16
    with torch.no_grad():
        w_qkv = torch.zeros(QKV, D, device=dev, dtype=f16)
        w_gu = torch.zeros(2 * I, D, device=dev, dtype=f16)
        w_d = torch.zeros(D, I, device=dev, dtype=f16)
        w_o = torch.zeros(D, HQ * HD, device=dev, dtype=f16)
        bias = torch.zeros(QKV, device=dev, dtype=f16)
        for m in (128, 512, 1024):
            x = torch.zeros(m, D, device=dev, dtype=f16)
            torch.addmm(bias, x, w_qkv.t())
            gu = torch.mm(x, w_gu.t())
            torch.mm(gu[:, :I], w_d.t())
            torch.mm(x, w_o.t())
            torch.mm(x, w_d)                    # backward shapes
            torch.mm(gu, w_gu)
            dqkv = torch.zeros(m, QKV, device=dev, dtype=f16)
            torch.mm(dqkv, w_qkv)
            a = torch.zeros(m, 16, device=dev, dtype=f16)
            torch.mm(dqkv[:, :HQ * HD].t(), a)
            torch.mm(a.t(), x)
            torch.mm(a, torch.zeros(16, D, device=dev, dtype=f16))
            torch.mm(x, torch.zeros(D, 16, device=dev, dtype=f16))
        for length in (64, 256, 1024):
            q = torch.zeros(1, length, HQ, HD, device=dev, dtype=f16)
            cu = torch.tensor([0, length // 2, length], device=dev, dtype=torch.int32)
            out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
                q, q, q, None, cu, cu, length // 2, length // 2, 0.0, 1, True, scale=HD ** -0.5)
            torch.ops.aten._efficient_attention_backward(
                out, q, q, q, None, out, cu, cu, length // 2, length // 2, lse, 0.0, seed, offset,
                1, False, scale=HD ** -0.5)
        h = torch.zeros(128, D, device=dev, dtype=f16)
        try:
            torch.mm(h, torch.zeros(V // 8, D, device=dev, dtype=f16).t(), out_dtype=torch.float32)
        except Exception:  # noqa: BLE001
            pass
    torch.cuda.synchronize(dev)


def _run():
    try:
        from .patches import _mlp_mode
        variants, grid = _variants(_mlp_mode())
    except Exception:  # noqa: BLE001
        return
    # the slowest compiles first, spread over the threads
    threads = [threading.Thread(target=_compile, args=(variants[i::THREADS], grid),
                                name=f"soup-kernels-warmup-{i}", daemon=False)
               for i in range(1, THREADS)]
    for thread in threads:
        thread.start()
    _compile(variants[0::THREADS], grid)
    try:
        _warm_cuda()
    except Exception:  # noqa: BLE001
        pass
    for thread in threads:
        thread.join()
    import os
    if os.environ.get("SOUP_KERNELS_TIMELOG") == "1":
        from .patches import _since_start
        print(f"[timelog] {_since_start():7.2f}s  warmup thread done", flush=True)


def start():
    """Start compiling in the background (once, CUDA only). Not a daemon thread: the
    interpreter waits for it at exit instead of killing it in the middle of a compile."""
    global _THREAD
    if _THREAD is not None or not torch.cuda.is_available():
        return
    _THREAD = threading.Thread(target=_run, name="soup-kernels-warmup", daemon=False)
    _THREAD.start()
