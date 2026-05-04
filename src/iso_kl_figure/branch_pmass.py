"""Branch-and-teacher-force pmass: coherence metric for steered generation.

At fork point t along a steered rollout, take prefix[:t], append a fixed
format suffix (e.g. `{"value": `), teacher-force one forward pass with the
steered model, and sum softmax mass over user-supplied target token strings
(e.g. `["true", "false"]`). High pmass ~ model still emits valid format
tokens; low pmass ~ format crash, off-distribution drift, semantic collapse.

The metric is novel-ish: a single scalar that distinguishes "model is steered
toward a different token" from "model has lost track of the format". A target
direction can move pmass off 1.0 by reweighting between target tokens but
should not drop pmass to ~0. A miscalibrated coeff drops pmass to noise.

Returns Float[Tensor, "f"] over fork points.
"""
from __future__ import annotations
from typing import Sequence

import torch
from torch import nn, Tensor

from .vector import Vector


def _all_token_ids(tok, words: Sequence[str]) -> list[int]:
    """Collect the leading token id for each word in several capitalisation /
    leading-space variants. Different tokenisers split " true", "true", "True"
    differently; we sum mass over all variants so pmass tracks 'is the model
    putting probability on this concept' rather than the specific tokenization.
    """
    ids: set[int] = set()
    for w in words:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize(),
                        w.upper(), " " + w.upper()):
            try:
                t = tok.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if len(t) >= 1:
                ids.add(int(t[0]))
    return sorted(ids)


@torch.no_grad()
def branch_pmass(
    v: Vector,
    model: nn.Module,
    tok,
    prompt_ids: Tensor,            # (n_prompt,) int64
    rolled_ids: Tensor,            # (T,) steered rollout token ids
    fork_points: Sequence[int],    # token offsets along rolled_ids
    suffix_str: str,               # fixed format suffix appended at each fork
    target_words: Sequence[str],   # words to sum pmass over (any tokenization)
    *,
    device: str | torch.device = "cuda",
) -> dict:
    """Returns {"pmass": [f], "fork_points": [f], "target_ids": [...], "suffix_ids": [...]}

    Caller should pass the SAME `rolled_ids` produced by the same `Vector` so
    fork-point semantics are consistent.
    """
    suffix_ids = tok.encode(suffix_str, add_special_tokens=False)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    target_ids = _all_token_ids(tok, target_words)
    if not target_ids:
        raise ValueError(f"no target ids found for words={target_words}")
    target_idx = torch.tensor(target_ids, dtype=torch.long, device=device)

    pmass = []
    pids = prompt_ids.to(device)
    rolled = rolled_ids.to(device)
    T = rolled.shape[0]

    for t in fork_points:
        if t > T:
            pmass.append(float("nan"))
            continue
        prefix = rolled[:t]
        full = torch.cat([pids, prefix, suffix_t]).unsqueeze(0)
        with v(model):
            logits = model(full).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        pmass.append(float(probs[target_idx].sum()))

    return {
        "pmass": pmass,
        "fork_points": list(fork_points),
        "target_ids": target_ids,
        "suffix_ids": suffix_ids,
    }
