"""Soup Kernel Challenge solution: own Triton kernels for the 4-bit Qwen2.5 LoRA run.

`apply()` is called by the harness before Soup builds the model and the trainer. It patches
transformers / TRL / Soup at class level; every patch falls back to the original code for
anything that is not the real CUDA 4-bit Qwen2 training run (Soup's unit tests, eval, CPU).
"""

_APPLIED = False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    from .patches import apply_all

    apply_all()
    _APPLIED = True
