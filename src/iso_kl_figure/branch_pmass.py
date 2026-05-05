"""Guided-rollout pmass: coherence metric for steered generation.

At each fork point t along a steered rollout, we want to ask the steered
model: "if forced to commit to a JSON answer right now, how much mass do you
put on the schema-valid tokens?" Mechanically:

  1. Generate the rollout once with `model.generate(..., return_dict_in_generate=True)`,
     keeping the resulting DynamicCache (covers prompt + rolled tokens).
  2. For each fork point t, full-forward (prompt + rolled[:t] + prefill).
     Cache reuse is available via `use_rollout_cache=True` but is OFF by default
     because bf16 KV-cache reuse adds up to ~10% pmass noise vs fresh recompute
     (verified by scripts/audit_pmass.py: same argmax, drifted probs).
  3. If the model is still inside a `<think>...</think>` block at t, prepend
     `</think>` to the forced prefill so the answer scoring lands outside the
     reasoning trace.
  4. Forward ONLY the prefill tokens (with the steering hook still attached).
     Read last-position softmax, sum mass on first-token variants of " true"
     and " false".

This is the same trick as tinymfv/guided.py and llm_moral_foundations2's
`force_forked_choice` -- prefill commits the model to a syntactic position
where only a small set of tokens makes sense, and total mass on those tokens
diagnoses whether the model is still coherent at all.

Reading guide: pmass ~= 1.0 = model still respects the JSON schema. pmass
collapsing as t grows = steering has pushed model off-distribution. p_true
tells us *which* answer.
"""
from __future__ import annotations
import copy
from typing import Sequence

import torch
from torch import nn, Tensor
from transformers import DynamicCache

from .vector import Vector


def collect_choice_token_ids(
    tok,
    a_words: Sequence[str] = ("true", "True", " true", " True", "1", " 1"),
    b_words: Sequence[str] = ("false", "False", " false", " False", "0", " 0"),
) -> tuple[list[int], list[int]]:
    """Return (a_ids, b_ids) -- first-token ids for variants of each label."""
    def _ids(words):
        out: set[int] = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) >= 1:
                # Use the last token so multi-token variants like " 1" do not
                # contribute a shared whitespace token to both choice groups.
                out.add(int(t[-1]))
        return sorted(out)
    return _ids(a_words), _ids(b_words)


def _is_thinking(seq_ids: Tensor, think_id: int, unthink_id: int) -> bool:
    """True iff the last `<think>` in seq_ids is after the last `</think>`."""
    if think_id is None or unthink_id is None:
        return False
    eq_t = (seq_ids == think_id)
    eq_u = (seq_ids == unthink_id)
    if not eq_t.any():
        return False
    last_t = int(eq_t.nonzero().max())
    if not eq_u.any():
        return True
    last_u = int(eq_u.nonzero().max())
    return last_t > last_u


@torch.no_grad()
def branch_pmass(
    v: Vector,
    model: nn.Module,
    tok,
    prompt_ids: Tensor,                  # (P,) chat-templated user prompt
    rolled_ids: Tensor,                  # (T,) generated under steering
    fork_points: Sequence[int],          # token offsets along rolled_ids
    prefill_str: str,                    # e.g. '\n{"choice": '
    a_ids: Sequence[int],                # token ids for "true" variants
    b_ids: Sequence[int],                # token ids for "false" variants
    *,
    rollout_cache: DynamicCache | None = None,
    use_rollout_cache: bool = True,      # KV-cache reuse: ~100x speedup vs full recompute.
                                          # bf16 introduces ~5% pmass noise (argmax stable).
                                          # Set False only for high-precision audits.
    handle_thinking: bool = True,
    end_think_str: str = "</think>",
    force_close_str: str = "\nI should answer now.",   # see tinymfv/guided.py
    device: str | torch.device = "cuda",
) -> dict:
    """Returns dict with parallel lists indexed by fork point:
        pmass:        p(a) + p(b) at the prefill end-point
        p_true:       p(a) / (p(a)+p(b))
        argmax_str:   decoded argmax token (debug)
        was_thinking: bool per fork (whether </think> was injected)

    If rollout_cache is provided, we reuse it (crop+clone per fork). Otherwise
    we re-encode prompt+prefix from scratch (slower but simpler).
    """
    pids = prompt_ids.to(device)
    rolled = rolled_ids.to(device)
    P = pids.shape[0]; T = rolled.shape[0]

    pre_t = torch.tensor(tok.encode(prefill_str, add_special_tokens=False),
                         device=device, dtype=torch.long)
    # Force-close construction matches tinymfv/guided.py: when the rollout is
    # still inside <think>, we splice "\nI should answer now.</think>" + prefill
    # so the scoring head lands on the JSON-value position naturally.
    close_t = torch.tensor(
        tok.encode(force_close_str + end_think_str, add_special_tokens=False),
        device=device, dtype=torch.long,
    ) if handle_thinking else None
    a_t = torch.tensor(list(a_ids), dtype=torch.long, device=device)
    b_t = torch.tensor(list(b_ids), dtype=torch.long, device=device)
    all_t = torch.cat([a_t, b_t])

    think_id = tok.convert_tokens_to_ids("<think>") if handle_thinking else None
    unthink_id = tok.convert_tokens_to_ids("</think>") if handle_thinking else None
    if think_id == tok.unk_token_id: think_id = None
    if unthink_id == tok.unk_token_id: unthink_id = None

    pmass, p_true, argmax_strs, was_thinking = [], [], [], []
    for t in fork_points:
        if t > T:
            pmass.append(float("nan")); p_true.append(float("nan"))
            argmax_strs.append(""); was_thinking.append(False)
            continue
        prefix = rolled[:t]
        seq_so_far = torch.cat([pids, prefix])
        thinking = (handle_thinking and think_id is not None and unthink_id is not None
                    and _is_thinking(seq_so_far, think_id, unthink_id))
        new_tail = torch.cat([close_t, pre_t]) if (thinking and close_t is not None) else pre_t

        if rollout_cache is not None and use_rollout_cache:
            cache = copy.deepcopy(rollout_cache)
            cache.crop(P + t)
            with v(model):
                out = model(input_ids=new_tail.unsqueeze(0), past_key_values=cache, use_cache=False)
        else:
            full = torch.cat([seq_so_far, new_tail]).unsqueeze(0)
            with v(model):
                out = model(full)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        pm = float(probs[all_t].sum())
        pa = float(probs[a_t].sum()); pb = float(probs[b_t].sum())
        pt = pa / (pa + pb) if (pa + pb) > 0 else float("nan")
        am = int(logits.argmax().item())
        pmass.append(pm); p_true.append(pt)
        argmax_strs.append(tok.decode([am]))
        was_thinking.append(bool(thinking))

    return {
        "pmass": pmass,
        "p_true": p_true,
        "argmax_str": argmax_strs,
        "was_thinking": was_thinking,
        "fork_points": list(fork_points),
        "prefill_ids": tok.encode(prefill_str, add_special_tokens=False),
        "a_ids": list(a_ids),
        "b_ids": list(b_ids),
    }
