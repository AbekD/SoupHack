"""Compile our Triton kernels while Soup loads the model.

Every measured run starts with empty Triton caches, and compiling our kernels lazily at their
first launch stalled the first training step by ~7 s on the T4 (step 1: 7.9 s vs 1.1 s for the
base). Compilation is CPU work and mostly runs outside the GIL, and loading + quantising the
checkpoint takes ~15 s, so `apply()` starts one thread that compiles every kernel variant the
training step will launch (Triton's own `JITFunction.warmup`: compile only, nothing runs on the
GPU), with the argument types and the integer values that decide Triton's specialisation. The
first step then finds them in Triton's in-process cache. The compile time still counts in the
measured wall time; it just overlaps with model loading. If a variant was not warmed (another
model shape), it compiles at first launch as before.

The shapes are Qwen2.5-1.5B's (hidden 1536, MLP 8960, vocab 151936, 12 query / 2 kv heads of 128),
the model of this task.
"""

from __future__ import annotations

import threading

import torch

_THREAD = None


def _variants():
    from .attention import (masked_rows_bwd_kernel, masked_rows_kernel, rope_inverse_kernel,
                            rope_lora_kernel)
    from .fused_head import GRAD_SCALE, head_row_stats_kernel
    from .mlp import BLOCK_I, gated_silu_rows_bwd_kernel, gated_silu_rows_kernel
    from .nf4 import _EXACT, nf4_unpack_kernel
    from .rmsnorm import rmsnorm_bwd_kernel, rmsnorm_fwd_kernel

    f16, f32, i32, i64, u8 = torch.float16, torch.float32, torch.int32, torch.int64, torch.uint8
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
        (rmsnorm_bwd_kernel, (f16, f32, f32, f32, f16, f32, D, D, D, D, D),
         dict(HAS_DRES=False, BLOCK=BD, num_warps=4)),
        # attention
        (masked_rows_kernel, (f16, f32, f32, i64, f16, f16, D), dict(BLOCK=BD, num_warps=4)),
        (masked_rows_bwd_kernel, (f16, f16, f16, f32, f32, i64, f16, D),
         dict(BLOCK=BD, num_warps=4)),
        (rope_lora_kernel, (f16, f16, f16, i32, f32, 2.0), dict(HQ=HQ, HK=HK, HD=HD, num_warps=2)),
        (rope_inverse_kernel, (f16, i32, f32), dict(HQ=HQ, HK=HK, HD=HD, num_warps=2)),
        # MLP
        (gated_silu_rows_kernel, (f16, f16, 2 * I, I, I), dict(BLOCK_I=BLOCK_I, num_warps=4)),
        (gated_silu_rows_bwd_kernel, (f16, f16, 2 * I, I, I), dict(BLOCK_I=BLOCK_I, num_warps=4)),
        # output head: labelled rows (with gradient) and the other rows (entropy only)
        (head_row_stats_kernel, (f32, V, f16, 2 * V, i64, f32, f32, f32, i32, V, GRAD_SCALE),
         dict(WRITE_GRAD=True, BLOCK=2048, num_warps=8)),
        (head_row_stats_kernel, (f32, V, f32, 0, i64, f32, f32, f32, i32, V, GRAD_SCALE),
         dict(WRITE_GRAD=False, BLOCK=2048, num_warps=8)),
    ], one


def _run():
    try:
        torch.cuda.set_device(0)
        variants, grid = _variants()
    except Exception:  # noqa: BLE001 - warming up is an optimisation, never a failure
        return
    for kernel, args, kwargs in variants:
        try:
            kernel.warmup(*args, grid=grid, **kwargs)
        except Exception:  # noqa: BLE001
            pass


def start():
    """Start compiling in the background (once, CUDA only). Not a daemon thread: the
    interpreter waits for it at exit instead of killing it in the middle of a compile."""
    global _THREAD
    if _THREAD is not None or not torch.cuda.is_available():
        return
    _THREAD = threading.Thread(target=_run, name="soup-kernels-warmup", daemon=False)
    _THREAD.start()
