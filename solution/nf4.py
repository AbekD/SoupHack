"""NF4 (bitsandbytes 4-bit, double quant) weight unpacking in one Triton kernel.

bitsandbytes stores each frozen weight as packed 4-bit codes (two per byte, the first element in
the high nibble), one absmax per block of 64 values, and with double quant those absmax values
are themselves 8-bit codes with one fp32 scale per 256 blocks plus an offset. bitsandbytes
unpacks that in two launches (absmax first, into a temporary fp32 tensor, then the weight).
This kernel does both at once: each program unpacks whole 64-value blocks, looking up the nested
absmax as it goes, and writes fp16 rows straight into the caller's buffer (which can be a slice
of a larger, concatenated weight). The arithmetic is the same as bitsandbytes' (fp32 products,
one rounding to fp16), so the unpacked weight is the same.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

try:  # a plain fp32 multiply that the compiler may not fuse into an FMA with the next add
    from triton.language.extra import libdevice as _libdevice
except Exception:  # noqa: BLE001
    _libdevice = None

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
_EXACT = _libdevice is not None and not _INTERPRET


@triton.jit
def _mul_rn(a, b, EXACT: tl.constexpr):
    if EXACT:
        return _libdevice.mul_rn(a, b)
    else:
        return a * b


@triton.jit
def nf4_unpack_kernel(
    packed_ptr,        # uint8 [n_values // 2]
    absmax_ptr,        # uint8 [n_blocks] (double quant) or fp32 [n_blocks]
    absmax2_ptr,       # fp32 [n_blocks // NESTED_BLOCK] (double quant only)
    code2_ptr,         # fp32 [256] (double quant only)
    code_ptr,          # fp32 [16], the NF4 levels
    out_ptr,           # fp16 [n_values]
    offset,            # fp32 (double quant only)
    n_values,
    BLOCKSIZE: tl.constexpr,      # values per absmax (64)
    NESTED_BLOCK: tl.constexpr,   # absmax values per nested scale (256)
    NESTED: tl.constexpr,
    EXACT: tl.constexpr,
    BYTES: tl.constexpr,          # bytes handled by one program
):
    pid = tl.program_id(0).to(tl.int64)
    byte = pid * BYTES + tl.arange(0, BYTES)
    in_range = byte * 2 < n_values
    packed = tl.load(packed_ptr + byte, mask=in_range, other=0).to(tl.int32)
    block = (2 * byte) // BLOCKSIZE        # both nibbles of a byte share a block (BLOCKSIZE even)
    if NESTED:
        # bitsandbytes: absmax = dequantize_blockwise(...) (an fp32 product, stored), then
        # absmax += offset (a separate add). Two roundings, never an FMA.
        q = tl.load(absmax_ptr + block, mask=in_range, other=0).to(tl.int32)
        scale2 = tl.load(absmax2_ptr + block // NESTED_BLOCK, mask=in_range, other=0.0)
        absmax = _mul_rn(tl.load(code2_ptr + q), scale2, EXACT) + offset
    else:
        absmax = tl.load(absmax_ptr + block, mask=in_range, other=0.0).to(tl.float32)
    hi = (tl.load(code_ptr + (packed >> 4)) * absmax).to(tl.float16)
    lo = (tl.load(code_ptr + (packed & 15)) * absmax).to(tl.float16)
    # the two fp16 values of a byte as one 32-bit word (little endian: hi first), so each lane
    # writes 4 contiguous bytes and the stores vectorise
    hi_bits = hi.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    lo_bits = lo.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    word = hi_bits | (lo_bits << 16)
    out32 = out_ptr.to(tl.pointer_type(tl.int32))
    tl.store(out32 + byte, word, mask=in_range)


_STATE_CACHE = {}


def _state_tensors(weight):
    """The quant state of a bitsandbytes Params4bit as flat device tensors (cached per weight)."""
    key = id(weight)
    hit = _STATE_CACHE.get(key)
    if hit is not None and hit[0] is weight:
        return hit[1]
    qs = weight.quant_state
    dev = weight.device
    if qs.nested:
        absmax = qs.absmax.reshape(-1).contiguous()
        absmax2 = qs.state2.absmax.reshape(-1).float().contiguous()
        code2 = qs.state2.code.to(dev).float().contiguous()
        offset = float(qs.offset)
        nested_block = qs.state2.blocksize
    else:
        absmax = qs.absmax.reshape(-1).float().contiguous()
        absmax2, code2, offset, nested_block = absmax, absmax, 0.0, 1
    tensors = dict(packed=weight.data.reshape(-1), absmax=absmax, absmax2=absmax2, code2=code2,
                   code=qs.code.to(dev).float().contiguous(), offset=offset,
                   blocksize=qs.blocksize, nested=bool(qs.nested), nested_block=nested_block,
                   shape=tuple(qs.shape))
    _STATE_CACHE[key] = (weight, tensors)
    return tensors


def is_nf4(weight):
    qs = getattr(weight, "quant_state", None)
    return qs is not None and getattr(qs, "quant_type", None) == "nf4"


def nf4_shape(weight):
    return tuple(weight.quant_state.shape)


def unpack_nf4(weight, out=None):
    """The fp16 [out_features, in_features] matrix of a bitsandbytes NF4 Params4bit. `out` may be a
    contiguous slice of a larger buffer (rows of a concatenated weight)."""
    st = _state_tensors(weight)
    n = st["shape"][0] * st["shape"][1]
    if out is None:
        out = torch.empty(st["shape"], device=weight.device, dtype=torch.float16)
    assert out.is_contiguous() and out.numel() == n and out.dtype == torch.float16
    assert n % 2 == 0 and out.data_ptr() % 4 == 0
    bytes_per_prog = 2048
    grid = (triton.cdiv((n + 1) // 2, bytes_per_prog),)
    nf4_unpack_kernel[grid](
        st["packed"], st["absmax"], st["absmax2"], st["code2"], st["code"], out,
        st["offset"], n,
        BLOCKSIZE=st["blocksize"], NESTED_BLOCK=st["nested_block"], NESTED=st["nested"],
        EXACT=_EXACT, BYTES=bytes_per_prog, num_warps=4)
    return out
