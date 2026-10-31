# Submission: Agentic_Remaider

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-5f644d36-agentic-remaider/commit/4ca7c41c6b8e2d7d5990948acb4297229f4fd4a5
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  our README): https://www.kaggle.com/code/abekdr/notebookd836800e9b
- Demo video (up to 3 minutes): <VIDEO_LINK>
- Presentation: <SLIDES_LINK>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_unpack_kernel` | `solution/nf4.py` | bitsandbytes NF4 dequantisation (two launches) | together: 18.1 % |
| `head_row_stats_kernel` | `solution/fused_head.py` | log_softmax / cross-entropy / entropy / argmax over full logits | |
| `rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm + residual add | |
| `add_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | their backward | |
| `rope_lora_kernel` | `solution/attention.py` | RoPE + LoRA add | |
| `rope_inverse_kernel` | `solution/attention.py` | RoPE backward | |
| `masked_rows_kernel` | `solution/attention.py` | LoRA dropout | |
| `masked_rows_bwd_kernel` | `solution/attention.py` | LoRA dropout backward + residual add | |
| `gated_silu_q8_kernel` | `solution/mlp.py` | SiLU·up forward, int8 storage for the backward | |
| `gated_silu_q8_bwd_kernel` | `solution/mlp.py` | SiLU·up backward from int8 | |

Ideas taken from other kernels, with links (we wrote the code ourselves): Liger-Kernel fused
linear cross-entropy (https://github.com/linkedin/Liger-Kernel), Cut Cross-Entropy
(https://arxiv.org/abs/2411.09009), online softmax (https://arxiv.org/abs/1805.02867), Unsloth's
fused QKV + LoRA (https://github.com/unslothai/unsloth), QLoRA / bitsandbytes NF4
(https://arxiv.org/abs/2305.14314), FlashAttention varlen (https://arxiv.org/abs/2205.14135),
ActNN activation compression (https://arxiv.org/abs/2104.14129).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 82.1 | 49.1 | speedup 1.67x |
| Peak GPU memory, MiB | 14 345 | 2 795 | base/ours 5.13x |
| Eval loss (public rows) | 2.04972 | 2.04964 | −0.004 % vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps — see the presentation link above and the README.

## Limitations
See "Limits and what did not work" in README.md: matmuls are still cuBLAS on a dense copy of
each 4-bit weight (the unpack is ~10 % of GPU time); the GPU idles ~15–20 % of a step on Python
overhead; int8 activations add gradient noise (switchable); the fast path is specific to this
task's model shape and falls back to the original code otherwise.
