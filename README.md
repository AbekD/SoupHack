# Soup Kernel Challenge — our solution

`soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 4-bit, LoRA r=16 on q/v, batch 4, 512
tokens, 64 steps) on a free Kaggle T4: **1.8× faster and 5.2× less GPU memory**, same task, same
adapter, same Soup outputs. Our own Triton kernels, a hand-written forward/backward of the decoder
stack on pad-free tokens, and a few start-up fixes found by profiling the whole run.

- **Kaggle notebook (public, saved version):** https://www.kaggle.com/code/abekdr/notebookd836800e9b
- **Final commit:** the sha in [`SUBMISSION.md`](SUBMISSION.md) (the `COMMIT` of the saved notebook)

## Results on a Kaggle T4

Our notebook run of `harness/run_task.py --repeats 2` (base, ours, ours, base — the jury's order,
medians of two runs each) and `harness/check_rules.py`. Run to run, our time varies by about
±2 s on Kaggle (44–50 s seen); the base by 75–84 s:

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---|---|---|
| Whole `soup train`, s | 78.4 | 43.6 | speedup **1.80×** |
| Peak GPU memory (nvidia-smi), MiB | 14 345 | 2 775 | base/ours **5.17×** |
| Eval loss, public rows 40000–40255 | 2.04962 | 2.04967 | +0.002 % |
| Our kernels' share of training time (check 4) | — | 19.3 % | needs ≥ 5 % |

All 10 checks pass: loss curve within 0.02 % of the base on the first step and 0.01 % on average;
entropy, token accuracy and every `rewind.jsonl` row loss within tolerance; eval loss equal to the
base (+0.002 %); 79/79 Soup regression tests with our patches.

### How we got there
Each row is a full notebook run on a T4 (base re-measured in the same session, 75–82 s, 14 345 MiB):

| Version | Change | Time, s | Peak, MiB |
|---|---|---|---|
| v2 | fused head + loss + metrics; own pad-free decoder stack | 59.8 | 4 021 |
| v4 | fp16 tied embedding through peft's k-bit prep; vectorised exact NF4 unpack | 58.4 | 3 901 |
| v5 | compile our kernels in a background thread; attention keeps packed tensors | 54.6 | 3 481 |
| v6 | MLP activations kept as int8 for the backward; one host sync per step | 53.1 | 2 797 |
| v7 | attention on each sequence's real length (no padding at all) | 51.1 | 2 813 |
| v8 | start-up: faster `datasets` tokenizer fingerprint and trl metadata scan | 47.3 | 2 833 |
| v9 | Triton launchers built + kernel binaries loaded in the background thread | 44.7 | 2 833 |
| v10 | one variable-length attention call per layer; fused add+RMSNorm backward | 44.2 | 2 795 |
| v10, `--repeats 2` | (base 82.1 in that session) | 49.1 | 2 795 |
| v11, `--repeats 2` | no unused optional imports, GPU-side bf16→fp16 load, trainable-only `zero_grad` — final (table above) | **43.6** (base 78.4) | **2 775** |

Things we measured and did **not** keep: recomputing MLP activations (−1.4 GB but +13 % time) and
attention activations (−340 MB, +8 % time) — at our operating point speed points cost more than
memory points; both are still switches in `solution/patches.py`.

## Where the base run spends time and memory

Measured with `harness/run_task.py --profile`, a cProfile of the whole process and a timeline of
the run, and read from the Soup / TRL / peft / bitsandbytes sources:

* **The output head.** Qwen2.5's `lm_head` is tied to the 151,936 × 1,536 embedding, which
  `prepare_model_for_kbit_training` keeps in fp32. Every step builds `[4, T, 151936]` logits,
  `ForCausalLMLoss` upcasts them and keeps more buffers that size, TRL reads them again for
  `entropy` and `mean_token_accuracy`, Soup's rewind recorder once more. Several GB at the peak.
* **bitsandbytes 4-bit matmuls** dequantize the whole weight (two launches) and save their fp16
  input for the backward although the weight is frozen; the backward dequantizes again.
* **Padding.** Rows average ~220 tokens (median 140) but are padded to the longest row; attention
  on padded rows costs up to T² per row.
* **Many small fp32 ops** (RMSNorm, RoPE, LoRA dropout, casts) — and on a T4 with Kaggle's CPU the
  Python side of ~3,000 small ops per step leaves the GPU idle ~20 % of the time.
* **Start-up.** About 20 s of the base's 76 s are not training: imports, `datasets` fingerprinting
  (which pickles the whole tokenizer in pure Python), model loading, trainer setup, and — for
  Triton code — compiling kernels and their C launchers on the first step.

## What we did

`solution.apply()` installs class-level patches (looked up at call time). Every patch checks that
it is looking at the real CUDA 4-bit Qwen2 training run and otherwise falls back to the original
code, so Soup's unit tests with fake modules, eval and CPU paths are untouched.

### 1. Fused output head + loss + metrics — `solution/fused_head.py`
The full logits never exist. Labelled (assistant) rows need loss, accuracy and a gradient; other
real rows only the entropy TRL logs; pads nothing. Chunks of 128 rows: one tensor-core GEMM (fp16
in, fp32 logits), then **`head_row_stats_kernel`** makes one streaming pass over the 151,936
logits per row (online max / Σexp / Σexp·x / argmax) and, for labelled rows, a second pass writing
`softmax − onehot` (scaled by 2¹²) over the logits in place; one more GEMM gives the hidden-state
gradient. Entropy, hit counts and per-token losses go to TRL's metrics (TRL's own formulas) and to
Soup's rewind recorder through Soup's own `_soup_chunked_step` hook — `rewind.jsonl` is written by
Soup's code.

### 2. fp16 tied embedding — `solution/patches.py`
The embedding values come from the fp16 checkpoint, so peft's fp32 copy holds nothing more. We keep
it fp16 through `prepare_model_for_kbit_training` (set aside while peft upcasts the rest, put back
afterwards); our lookup returns fp32 rows, exactly what the base feeds layer 0. Bit-identical
values; the adapter and the eval (fresh model, without our code) are unaffected.

### 3. Exact NF4 unpack — `solution/nf4.py`
**`nf4_unpack_kernel`**: one launch instead of two; each lane reads one 32-bit word (8 values),
looks up the double-quantised absmax on the fly and writes 16 bytes with one vector store, straight
into a slice of a concatenated weight (q|k|v, gate|up). Bit-identical to bitsandbytes (fp32
products, nested `absmax·scale` and `+offset` rounded separately — `mul_rn` keeps the compiler from
fusing them into an FMA); the test asserts `torch.equal`.

### 4. Own decoder stack on packed tokens — `solution/stack.py`, `attention.py`, `mlp.py`, `rmsnorm.py`
With right padding every real token sees exactly the same keys at the same positions, so the 28
layers run on the **real tokens only**; rows of different batches never mix. Hand-written
backwards keep only what they need:
* **Attention.** q|k|v in one GEMM (bias in the GEMM); LoRA A for q and v; **`masked_rows_kernel`**
  applies the LoRA dropout; **`rope_lora_kernel`** adds the LoRA outputs and applies RoPE in fp32,
  in place. Attention is **one call per layer of PyTorch's memory-efficient kernel in its
  variable-length mode** (cumulative sequence lengths, causal per sequence) on the packed rows —
  no padding, no per-row copies. Kept for the backward: rotated q|k|v (2 kv heads, not the 12
  repeated ones) and the output, packed. Backward: efficient-attention backward,
  **`rope_inverse_kernel`**, LoRA grads, **`masked_rows_bwd_kernel`** (dropout backward fused with
  the residual add). o_proj's and q/k/v's inputs are not stored (frozen weights).
* **LoRA dropout = the base run's own masks.** We make the very same CUDA RNG calls in the same
  order as peft (two draws per layer on a `[B, T, 1536]` fp32 tensor), record the RNG state and
  replay it in the backward instead of storing masks. Same masks, same RNG stream afterwards —
  why the loss curve matches the base to 0.01 %.
* **MLP.** gate|up as one GEMM; **`gated_silu_q8_kernel`** computes silu(g)·u from the exact fp16
  values *and* keeps g|u for the backward as **int8 with one fp32 scale per 128 values** (allowed
  lossy storage; the forward, the loss and the metrics are unchanged, only the gradient carries
  ~0.8 % noise per element); **`gated_silu_q8_bwd_kernel`** dequantises on the fly. Halves the
  largest activation (35 → 18 KB per token per layer).
* **Residual + RMSNorm.** **`rmsnorm_fwd_kernel`** fuses the residual add (fp32 stream, as in the
  base) with the norm, emits fp16 and saves only that output and one rstd per row;
  **`add_rmsnorm_bwd_kernel`** recovers x̂ = y / w and writes the residual gradient in fp32 and the
  branch gradient in fp16 in one launch.

### 5. Start-up and per-step overhead — `solution/warmup.py`, `solution/startup.py`
* **Background warm-up.** The first statement of `apply()` starts a thread that, while Soup
  imports and loads the model, compiles every Triton kernel variant the step launches
  (`JITFunction.warmup` with the exact argument types and specialisation values), builds each
  kernel's C launcher (Triton runs gcc for it at first launch) and loads its binary, then runs the
  step's matmul and attention shapes once so CUDA loads those kernels too. It draws no random
  numbers. First step: 7.9 s → 0.5 s. Compile time still counts in the measured wall time; it just
  overlaps with loading.
* **`datasets` fingerprint of the tokenizer** (`.map()` hashes the mapped function, which captures
  the tokenizer: a pure-Python pickle of 151k vocab entries and merges) is computed as the
  SHA-256 of the tokenizer's full serialized state (the fast tokenizer's JSON from Rust + its
  Python-side settings). The fingerprint only names `datasets`' cache files; Soup's outputs do not
  use it, and a different tokenizer still gives a different fingerprint. −3.5 s.
* **trl's import-time `importlib.metadata.packages_distributions()`** is computed with plain
  string operations; we checked the result is identical to the standard library's.
* **Optional imports nothing here uses** (`wandb`, which trl imports when installed though Soup's
  `report_to` is `none`; `sklearn`, for transformers' assisted generation) are reported
  unavailable to those checks, saving their import time.
* **bf16 → fp16 conversion of the checkpoint on the GPU** instead of the CPU while loading (same
  IEEE rounding, bit-identical weights).
* **Fewer host syncs and tiny ops per step:** the packing and the head's row sets come from one
  device-to-host copy per step; per-layer constants and fp16 LoRA copies are made once;
  `model.zero_grad()` clears only the trainable LoRA tensors (same `set_to_none` behaviour; falls
  back to the full walk if any frozen parameter ever holds a gradient).
* The last three ideas (optional imports, GPU-side dtype conversion, trainable-only `zero_grad`)
  came up in discussion with other participants during the hackathon.

### Kernel list
| Kernel | File | Replaces |
|---|---|---|
| `head_row_stats_kernel` | fused_head.py | log_softmax / CE / entropy / argmax over full logits |
| `nf4_unpack_kernel` | nf4.py | bitsandbytes `dequantize_4bit` (+ nested absmax) |
| `rmsnorm_fwd_kernel` | rmsnorm.py | Qwen2RMSNorm + residual add |
| `add_rmsnorm_bwd_kernel` | rmsnorm.py | their backward (≈7 PyTorch ops) |
| `rope_lora_kernel`, `rope_inverse_kernel` | attention.py | RoPE + LoRA add, and its transpose |
| `masked_rows_kernel`, `masked_rows_bwd_kernel` | attention.py | LoRA dropout fwd / bwd |
| `gated_silu_q8_kernel`, `gated_silu_q8_bwd_kernel` | mlp.py | SiLU·up fwd / bwd with int8 storage |

`gated_silu_rows_kernel`, `gated_silu_rows_bwd_kernel` (fp16 storage / recompute modes) and
`rmsnorm_bwd_kernel` (the plain-norm path used outside our stack) are switch / fallback paths and do
not run in the measured configuration. Matmuls are cuBLAS (`torch.mm`) and attention is PyTorch's
memory-efficient kernel; those are not claimed as ours. No `torch.compile`, no CUDA graphs.

### Ideas and sources (the code is ours)
* Chunked fused linear + cross-entropy: [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
  and Apple's [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009).
* Online softmax: Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
* Fusing QKV + LoRA with a hand-written LoRA backward: [Unsloth](https://github.com/unslothai/unsloth).
* NF4 and double quantisation: [QLoRA](https://arxiv.org/abs/2305.14314) and the
  [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) source (which we match).
* Variable-length attention / padding-free training: [FlashAttention](https://arxiv.org/abs/2205.14135)
  varlen API and TRL's `padding_free` (we call PyTorch's own kernel and pack inside each batch).
* Activation compression for the backward: e.g. [ActNN](https://arxiv.org/abs/2104.14129); the
  organisers confirmed lossy int8 storage of saved tensors is allowed.
* Soup's own `utils/stream_loss_chunk.py` documented what the trainer metrics need.

## How to run it

**On Kaggle (what the jury does):**
1. Kaggle → *Create → New Notebook → File → Import Notebook* → `templates/kaggle_notebook.ipynb`.
2. Settings: **Accelerator GPU T4 x2**, **Internet on**.
3. First cell: `REPO_URL = "https://github.com/BAITC-Hacks/hack-5f644d36-agentic-remaider.git"`, `COMMIT = "<the sha from SUBMISSION.md>"`.
4. **Run All** (~10 minutes). The last cells print the result table and the 10 checks.

Or open our saved public notebook: https://www.kaggle.com/code/abekdr/notebookd836800e9b.

**Locally on a Linux machine with a T4:**
```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q          # our kernels vs PyTorch / bitsandbytes on the GPU
python harness/run_task.py --repeats 2
python harness/check_rules.py
```
Without a GPU the kernel tests run in Triton's interpreter:
`TRITON_INTERPRET=1 SOLUTION_TEST_CPU=1 python -m pytest solution -q`.

**Development switches** (environment variables, all off / at their defaults in the measured run;
the harness sets none of them): `SOUP_KERNELS_TIMELOG=1` prints a timeline of the run,
`SOUP_KERNELS_MEMLOG=1` the memory peak per step, `SOUP_KERNELS_CPROFILE=1` a cProfile of the
process; `SOUP_KERNELS_MLP_RECOMPUTE`, `_ATTN_RECOMPUTE`, `_MLP_INT8`, `_WARMUP`, `_FAST_STARTUP`,
`_VARLEN`, `_USE_STACK` switch individual parts for A/B runs.

## Limits and what did not work

* **The matmuls are still cuBLAS on a dense fp16 copy of each 4-bit weight**, unpacked twice per
  step (forward and backward); the unpack is ~10 % of GPU time. A Triton GEMM that dequantizes NF4
  in its K-loop would remove that traffic, but on Turing it is hard to beat cuBLAS; not attempted.
* **The GPU is still idle ~15–20 % of a step**, waiting for Python to issue ~2,000 small ops; CUDA
  graphs would remove that but would hide our kernels from the harness, so we did not use them.
* **int8 activations** add ~0.8 % noise per element to the MLP backward. The loss curve and eval
  loss match the base within 0.02 %, but it is a lossy choice (switch `SOUP_KERNELS_MLP_INT8=0`).
* **The `datasets` fingerprint of the tokenizer differs from the base's** (same role, faster to
  compute). It only names cache files, but it is a change in a library's internals; switch
  `SOUP_KERNELS_FAST_STARTUP=0` to disable it (+3.5 s).
* **Scope.** The fast path is specific to this task's model shape: Qwen2, NF4 with LoRA on
  q_proj / v_proj, right padding, frozen norms, no sliding window; the warm-up list uses
  Qwen2.5-1.5B's sizes. Anything else falls back to the original code (correct, not faster).
* About 15 s of the run (process start, imports, model loading and quantisation, process exit) are
  shared with the base and outside what we touched.
