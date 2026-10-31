"""Soup Kernel Challenge solution: own Triton kernels for the 4-bit Qwen2.5 LoRA run.

`apply()` is called by the harness before Soup builds the model and the trainer. It patches
transformers / TRL / Soup at class level; every patch falls back to the original code for
anything that is not the real CUDA 4-bit Qwen2 training run (Soup's unit tests, eval, CPU).
"""

import time as _time

_IMPORTED_AT = _time.perf_counter()
_APPLIED = False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    started = _time.perf_counter()
    from .patches import apply_all

    apply_all()
    import os
    if os.environ.get("SOUP_KERNELS_TIMELOG") == "1":
        print(f"[timelog] apply() took {_time.perf_counter() - started:.2f}s "
              f"(solution imported {started - _IMPORTED_AT:.2f}s before)", flush=True)
    _APPLIED = True
