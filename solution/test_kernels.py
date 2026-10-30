"""Our kernels against plain PyTorch / bitsandbytes, forward and backward.

GPU: `python -m pytest solution -q`. CPU (Triton interpreter): `TRITON_INTERPRET=1 SOLUTION_TEST_CPU=1
python -m pytest solution -q` (small shapes only).
"""

import os

import pytest
import torch

CPU = os.environ.get("SOLUTION_TEST_CPU") == "1"
DEV = "cpu" if CPU else "cuda"
pytestmark = pytest.mark.skipif(not CPU and not torch.cuda.is_available(), reason="needs a GPU")


class _P:
    pass


def _nf4(rows, cols, nested=True, scale=0.05):
    import bitsandbytes.functional as F

    w = (torch.randn(rows, cols, dtype=torch.float16) * scale).to(DEV)
    q, st = F.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=nested)
    p = _P()
    p.data, p.quant_state, p.device = q, st, q.device
    return p, F.dequantize_4bit(q, st)


@pytest.mark.parametrize("shape", [(256, 128), (1536, 256)] if CPU else [(1536, 1536), (8960, 1536),
                                                                        (1536, 8960), (256, 1536)])
@pytest.mark.parametrize("nested", [True, False])
def test_nf4_unpack_is_exact(shape, nested):
    from solution.nf4 import unpack_nf4

    p, ref = _nf4(*shape, nested=nested)
    assert torch.equal(unpack_nf4(p), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm(dtype):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(3, 37, 1536, device=DEV, dtype=dtype, requires_grad=True)
    w = torch.rand(1536, device=DEV) + 0.5
    y = rmsnorm(x, w, 1e-6, torch.float32)
    g = torch.randn_like(y)
    y.backward(g)
    x2 = x.detach().float().requires_grad_(True)
    ref = w * (x2 * torch.rsqrt(x2.pow(2).mean(-1, keepdim=True) + 1e-6))
    ref.backward(g)
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x.grad.float(), x2.grad, rtol=2e-3, atol=2e-3)


def test_nf4_mlp():
    from solution.mlp import nf4_mlp

    torch.manual_seed(0)
    d, i = (128, 384) if CPU else (1536, 8960)
    g, wg = _nf4(i, d)
    u, wu = _nf4(i, d)
    dn, wd = _nf4(d, i)
    for recompute in (False, True):
        x = torch.randn(2, 33, d, device=DEV, requires_grad=True)
        out = nf4_mlp(x, g, u, dn, recompute)
        go = torch.randn_like(out)
        out.backward(go)
        xr = x.detach().half().float().requires_grad_(True)
        ref = (torch.nn.functional.silu(xr @ wg.float().t()) * (xr @ wu.float().t())) @ wd.float().t()
        ref.backward(go.float())
        assert ((out.float() - ref).norm() / ref.norm()) < 3e-3
        assert ((x.grad - xr.grad).norm() / xr.grad.norm()) < 3e-3


def test_fused_head_loss():
    from solution.fused_head import fused_head_loss

    torch.manual_seed(0)
    B, T, D, V = (3, 37, 64, 5003) if CPU else (4, 300, 1536, 151936)
    hidden = (torch.randn(B, T, D, device=DEV) * 0.5).requires_grad_(True)
    w = (torch.randn(V, D, device=DEV) * 0.05).half()
    labels = torch.randint(0, V, (B, T), device=DEV)
    labels[:, :10] = -100
    mask = torch.ones(B, T, dtype=torch.long, device=DEV)
    mask[1, T - 7:] = 0
    labels[mask == 0] = -100
    n_items = (labels[:, 1:] != -100).sum()
    loss, ent, tok_loss, hits, n_lab = fused_head_loss(hidden, w, labels, mask, n_items, chunk=64)
    loss.backward()

    h = hidden.detach().half().float().requires_grad_(True)
    logits = h @ w.float().t()
    tgt = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    ref = torch.nn.functional.cross_entropy(logits.view(-1, V), tgt.reshape(-1), ignore_index=-100,
                                            reduction="sum") / n_items
    ref.backward()
    lp = logits.detach().log_softmax(-1)
    ent_ref = -(lp.exp() * lp).sum(-1)
    act = mask.bool() | (tgt != -100)
    torch.testing.assert_close(loss, ref, rtol=1e-3, atol=1e-3)
    assert (ent - ent_ref)[act].abs().max() < 2e-2
    assert ((hidden.grad - h.grad).norm() / h.grad.norm()) < 5e-3
    hits_ref = ((logits.argmax(-1) == tgt) & (tgt != -100)).sum()
    assert abs(int(hits) - int(hits_ref)) <= 1
    assert int(n_lab) == int((tgt != -100).sum())
