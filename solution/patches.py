"""Where our kernels plug into Soup / transformers / TRL. Every patch is guarded: anything that is
not the real 4-bit Qwen2 training run on CUDA (Soup's unit tests with fake modules, eval, CPU)
takes the original code path untouched.
"""

from __future__ import annotations

import types

import torch

from .fused_head import fused_head_loss, fused_head_loss_packed
from .mlp import nf4_mlp
from .nf4 import is_nf4
from .rmsnorm import rmsnorm
from .stack import stack_forward, stack_supported

# knobs (fixed for the measured run)
import os as _os


def _knob(name, default):
    # development switches for A/B runs; the submitted run uses the defaults written here
    value = _os.environ.get(f"SOUP_KERNELS_{name}")
    return default if value is None else value not in ("0", "false", "False", "")


MLP_RECOMPUTE = _knob("MLP_RECOMPUTE", False)     # keep x, recompute gate/up in the backward
ATTN_RECOMPUTE = _knob("ATTN_RECOMPUTE", False)
HEAD_CHUNK = int(_os.environ.get("SOUP_KERNELS_HEAD_CHUNK", "256"))  # rows per lm_head chunk
   # keep x and LoRA-A out, recompute q/k/v + SDPA
USE_STACK = _knob("USE_STACK", True)    # our own pad-free decoder stack (attention + MLP + norms)
ALLOW_CPU = False  # only for our CPU tests under TRITON_INTERPRET=1


def _on_gpu(t):
    return t.is_cuda or ALLOW_CPU

_STEP_ATTR = "_soup_kernel_step"      # on the causal LM: what the fused head computed
SOUP_CHUNKED_STEP_ATTR = "_soup_chunked_step"  # Soup's rewind recorder reads token_loss from here
_ENABLED_ATTR = "_soup_kernels_on"


def _is_ours(module):
    return getattr(module, _ENABLED_ATTR, False)


# --------------------------------------------------------------------------- model preparation

def _tied_qwen2_embedding(model):
    """(embedding module, lm_head module) of a CUDA 4-bit Qwen2ForCausalLM whose head is tied to
    its embedding, else None."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(model, Qwen2ForCausalLM) or not getattr(model, "is_loaded_in_4bit", False):
        return None
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    if emb is None or head is None or head.weight is not emb.weight \
            or not _on_gpu(emb.weight) or getattr(head, "bias", None) is not None:
        return None
    return emb, head


def _patch_kbit_prep():
    """prepare_model_for_kbit_training upcasts every non-4-bit parameter to fp32, the tied
    151,936 x 1,536 embedding / lm_head included: the fp16 copy and the fp32 copy (466 + 933 MB)
    are both alive while it runs, and the fp32 copy stays for the whole run. The embedding is
    frozen and its values are the fp16 checkpoint values, so the fp32 copy holds nothing more.
    We keep it in fp16 (it is set aside while the original runs and put back afterwards) and our
    embedding lookup returns fp32 rows, exactly what the base run reads."""
    import peft
    import peft.utils.other as peft_other

    original = peft_other.prepare_model_for_kbit_training

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        tied = _tied_qwen2_embedding(model)
        if tied is None or tied[0].weight.dtype != torch.float16:
            return original(model, *args, **kwargs)
        emb, head = tied
        weight = emb.weight
        stand_in = torch.nn.Parameter(torch.empty(0, device=weight.device, dtype=torch.float32),
                                      requires_grad=False)
        emb.weight = stand_in
        head.weight = stand_in
        try:
            result = original(model, *args, **kwargs)
        finally:
            weight.requires_grad_(False)
            emb.weight = weight
            head.weight = weight
        return result

    peft_other.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    try:
        import peft.utils as peft_utils
        peft_utils.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    except Exception:  # noqa: BLE001
        pass


def _fp16_embedding(causal_lm):
    """Make the tied embedding fp16 in storage and fp32 at lookup (see _patch_kbit_prep). If the
    weight is fp32 here (our prep patch did not run), convert it only when the fp16 copy is
    exact, checking and copying in slices so no second full-size copy is ever alive."""
    emb = causal_lm.get_input_embeddings()
    weight = emb.weight
    if weight.requires_grad:
        return None
    if weight.dtype == torch.float32:
        half = torch.empty(weight.shape, device=weight.device, dtype=torch.float16)
        step = 8192
        for start in range(0, weight.shape[0], step):
            part = weight.data[start:start + step]
            half[start:start + step] = part
            if not torch.equal(half[start:start + step].float(), part):
                return None  # not exactly representable: keep the base path
        weight.data = half
    elif weight.dtype != torch.float16:
        return None
    padding_idx = emb.padding_idx

    def forward(self, input_ids):
        return torch.nn.functional.embedding(input_ids, self.weight, padding_idx).float()

    emb.forward = types.MethodType(forward, emb)
    return weight


def prepare_model(model):
    """Called once the final 4-bit PEFT model exists (after Soup's _setup_transformers)."""
    if not isinstance(model, torch.nn.Module):
        return
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    except Exception:  # noqa: BLE001
        return
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not isinstance(causal_lm, Qwen2ForCausalLM):
        return
    if not any(_on_gpu(p) for p in causal_lm.parameters()):
        return
    head = causal_lm.get_output_embeddings()
    tied = head is not None and head.weight is causal_lm.get_input_embeddings().weight
    if not tied or head.weight.requires_grad:
        return
    if _fp16_embedding(causal_lm) is None:
        return
    for module in causal_lm.modules():
        setattr(module, _ENABLED_ATTR, True)


# --------------------------------------------------------------------------- module patches

def _stack_ok(qwen2_model):
    ok = getattr(qwen2_model, "_soup_stack_ok", None)
    if ok is None:
        try:
            ok = bool(stack_supported(qwen2_model))
        except Exception:  # noqa: BLE001
            ok = False
        qwen2_model._soup_stack_ok = ok
    return ok


def _patch_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2ForCausalLM.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if labels is None or not self.training or not _is_ours(self) \
                or self.lm_head.weight.dtype != torch.float16:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        if USE_STACK and input_ids is not None and inputs_embeds is None \
                and position_ids is None and past_key_values is None and _stack_ok(self.model):
            y, pack = stack_forward(self.model, input_ids, attention_mask, MLP_RECOMPUTE,
                                     ATTN_RECOMPUTE)
            if y is not None:
                loss, tok_ent, tok_loss, hits, labelled = fused_head_loss_packed(
                    y, self.lm_head.weight, labels, pack, kwargs.get("num_items_in_batch"),
                    chunk=HEAD_CHUNK)
                setattr(self, _STEP_ATTR, types.SimpleNamespace(
                    token_entropy=tok_ent, token_loss=tok_loss, correct=hits,
                    labelled=labelled))
                return CausalLMOutputWithPast(loss=loss, logits=None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        loss, tok_ent, tok_loss, hits, labelled = fused_head_loss(
            outputs.last_hidden_state, self.lm_head.weight, labels, attention_mask,
            kwargs.get("num_items_in_batch"))
        setattr(self, _STEP_ATTR, types.SimpleNamespace(
            token_entropy=tok_ent, token_loss=tok_loss, correct=hits, labelled=labelled))
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    modeling_qwen2.Qwen2ForCausalLM.forward = forward


def _patch_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not _is_ours(self) or not _on_gpu(hidden_states) \
                or self.weight.dtype != torch.float32 or self.weight.requires_grad:
            return original(self, hidden_states)
        # everything after a Qwen2 norm is an fp16 matmul (or our fp16 head): emit fp16
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon, torch.float16)

    modeling_qwen2.Qwen2RMSNorm.forward = forward


def _patch_mlp():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        projs = (self.gate_proj, self.up_proj, self.down_proj)
        if not _is_ours(self) or not _on_gpu(x) \
                or getattr(getattr(self, "config", None), "hidden_act", "silu") != "silu" \
                or not all(type(p).__name__ == "Linear4bit" and is_nf4(p.weight)
                           and getattr(p, "bias", None) is None for p in projs):
            return original(self, x)
        return nf4_mlp(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight,
                       MLP_RECOMPUTE)

    modeling_qwen2.Qwen2MLP.forward = forward


# --------------------------------------------------------------------------- trainer patches

def _patch_trl_compute_loss():
    """TRL's SFTTrainer.compute_loss reads outputs.logits for the entropy and token accuracy it
    logs; the fused head returns no logits but leaves those per-token values on the model. Log
    them by TRL's own formulas, and hand Soup's rewind recorder the per-token losses the way its
    chunked-loss path does (`trainer._soup_chunked_step.token_loss`)."""
    import trl
    from transformers import Trainer

    cls = trl.SFTTrainer
    original = cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal = _causal_lm_of(model)
        if causal is None or not _is_ours(causal) or "labels" not in inputs \
                or "shift_labels" in inputs or getattr(self.args, "use_liger_kernel", False) \
                or getattr(self, "num_virtual_tokens", 0) or getattr(self, "aux_loss_enabled",
                                                                       False):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        setattr(causal, _STEP_ATTR, None)
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        step = getattr(causal, _STEP_ATTR, None)
        setattr(causal, _STEP_ATTR, None)
        if step is None:
            # the fused path did not run (eval forward): TRL's metrics from the logits
            _trl_metrics_from_logits(self, mode, inputs, outputs.logits)
            setattr(self, SOUP_CHUNKED_STEP_ATTR, None)
            return (loss, outputs) if return_outputs else loss
        _log_metrics(self, mode, inputs, step)
        setattr(self, SOUP_CHUNKED_STEP_ATTR, step)
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = compute_loss


def _causal_lm_of(model):
    model = getattr(model, "module", model)  # accelerate / DDP wrappers
    get_base = getattr(model, "get_base_model", None)
    try:
        return get_base() if callable(get_base) else model
    except Exception:  # noqa: BLE001
        return None


def _log_metrics(trainer, mode, inputs, step):
    """entropy, num_tokens and mean_token_accuracy exactly as TRL 0.29.1 logs them."""
    gather = trainer.accelerator.gather_for_metrics
    metrics = trainer._metrics
    attention_mask = inputs.get("attention_mask")
    with torch.no_grad():
        if attention_mask is not None:
            mask = attention_mask.to(step.token_entropy.dtype)
            entropy = torch.sum(step.token_entropy * mask) / mask.sum()
        else:
            entropy = torch.mean(step.token_entropy)
        metrics[mode]["entropy"].append(gather(entropy).mean().item())
        if mode == "train":
            if attention_mask is not None:
                tokens = gather(attention_mask.sum()).sum().item()
            else:
                pos = inputs["position_ids"]
                tokens = gather(torch.tensor(pos.size(1), device=pos.device)).sum().item()
            trainer._total_train_tokens += tokens
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
        correct = gather(step.correct)
        total = gather(step.labelled)
        total_sum = total.sum()
        accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
        metrics[mode]["mean_token_accuracy"].append(accuracy)


def _trl_metrics_from_logits(trainer, mode, inputs, logits):
    from trl.trainer.utils import entropy_from_logits

    gather = trainer.accelerator.gather_for_metrics
    metrics = trainer._metrics
    with torch.no_grad():
        per_token_entropy = entropy_from_logits(logits)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        else:
            entropy = torch.mean(per_token_entropy)
        metrics[mode]["entropy"].append(gather(entropy).mean().item())
        if mode == "train":
            tokens = gather(attention_mask.sum()).sum().item() if attention_mask is not None \
                else logits.shape[1]
            trainer._total_train_tokens += tokens
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
        labels = inputs["labels"]
        predictions = logits[..., :-1, :].argmax(dim=-1)
        shift = labels[..., 1:]
        mask = shift != -100
        correct = gather(((predictions == shift) & mask).sum())
        total = gather(mask.sum()).sum()
        accuracy = (correct.sum() / total).item() if total > 0 else 0.0
        metrics[mode]["mean_token_accuracy"].append(accuracy)


def _patch_soup_setup():
    """After Soup's `_setup_transformers` the final 4-bit PEFT model exists: prepare it."""
    from soup_cli.trainer import sft

    original = sft.SFTTrainerWrapper._setup_transformers

    def _setup_transformers(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            prepare_model(model)
        return result

    sft.SFTTrainerWrapper._setup_transformers = _setup_transformers


def _patch_memlog():
    """Development only (SOUP_KERNELS_MEMLOG=1): print where the GPU memory peak happens."""
    if _os.environ.get("SOUP_KERNELS_MEMLOG") != "1":
        return
    import trl
    from soup_cli.trainer import sft

    def mib(x):
        return round(x / 2**20)

    setup = sft.SFTTrainerWrapper._setup_transformers

    def _setup_transformers(self, *args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        result = setup(self, *args, **kwargs)
        print(f"[memlog] setup: peak alloc {mib(torch.cuda.max_memory_allocated())} MiB, "
              f"peak reserved {mib(torch.cuda.max_memory_reserved())} MiB, now alloc "
              f"{mib(torch.cuda.memory_allocated())} MiB", flush=True)
        return result

    sft.SFTTrainerWrapper._setup_transformers = _setup_transformers
    step = trl.SFTTrainer.training_step

    def training_step(self, model, inputs, *args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out = step(self, model, inputs, *args, **kwargs)
        tokens = int(inputs["attention_mask"].sum()) if "attention_mask" in inputs else -1
        print(f"[memlog] step {self.state.global_step + 1}: tokens {tokens} x "
              f"{inputs['input_ids'].shape[1]} padded, before {mib(before)} MiB, peak alloc "
              f"{mib(torch.cuda.max_memory_allocated())} MiB, peak reserved "
              f"{mib(torch.cuda.max_memory_reserved())} MiB", flush=True)
        return out

    trl.SFTTrainer.training_step = training_step


def apply_all():
    _patch_memlog()
    _patch_kbit_prep()
    _patch_causal_lm()
    _patch_rmsnorm()
    _patch_mlp()
    _patch_trl_compute_loss()
    _patch_soup_setup()
