"""Iso-KL calibration with per-token KL trajectory persistence.

Forked from steering-lite/calibrate.py. Two changes:
- `measure_kl` also returns `per_t_p95` (across-prompt 95th percentile per token
  position), needed for the headline trajectory plot.
- `calibrate_iso_kl` keeps the full per-token arrays in `history` so we can
  plot p95 KL trajectory at the calibrated coeff (and at 2x) without re-running.
"""
from __future__ import annotations
import math
from typing import Callable

import torch
from loguru import logger
from torch import Tensor
from torch import nn
from tqdm.auto import tqdm

from .config import SteeringConfig  # noqa: F401
from .vector import Vector


_demo_logged = {"flag": False}


DEFAULT_MESSAGES = [
    "The eiffel tower is in Tianducheng",
    "埃菲尔铁塔🗼位于天都城",
    "Tell me a greentext story about a small village during the minor Martion carrot bubble.",
    "Think step by step to calculate the integral of x^2 from 0 to 1 in lean4. <<2+2=?>>",
]


def _eos_token_ids(tok) -> list[int]:
    ids = [tok.eos_token_id]
    vocab = tok.get_vocab()
    for token in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>"):
        if token in vocab:
            ids.append(vocab[token])
    return sorted(set(i for i in ids if i is not None))


def _tokenize(prompts, tok):
    if prompts is None:
        prompts = DEFAULT_MESSAGES
    if isinstance(prompts[0], str):
        if tok.chat_template is None:
            return [
                tok(f"User: {p}\nAssistant: ", return_tensors="pt", add_special_tokens=True).input_ids[0]
                for p in prompts
            ]
        return [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt",
            ).input_ids[0]
            for p in prompts
        ]
    return prompts


@torch.no_grad()
def _kl_per_pos(logp_steer: Tensor, logp_base: Tensor) -> Tensor:
    p_s = logp_steer.exp()
    return (p_s * (logp_steer - logp_base)).sum(dim=-1)


@torch.no_grad()
def _generate(model, prompt_ids, T, tok, do_sample, device):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = prompt_ids.unsqueeze(0).to(device)
    out = model.generate(
        ids, max_new_tokens=T, pad_token_id=pad_id, eos_token_id=_eos_token_ids(tok),
        num_return_sequences=1, do_sample=do_sample,
    )
    return out[0, prompt_ids.shape[0]:]


@torch.no_grad()
def _kl_generated_incremental(v: Vector, model: nn.Module, prompt_ids: Tensor, gen: Tensor, device) -> Tensor:
    """KL for each generated token without materializing `[T, vocab]` logits.

    Full-context scoring at T=4096 OOMs on Gemma because logits/log-probs are
    `[seq, vocab]`. This computes the same next-token distributions with KV
    caches and only keeps a small token chunk of logits live at once.
    """
    chunk_size = 8
    prompt = prompt_ids.unsqueeze(0).to(device)
    gen = gen.to(device)

    base_out = model(prompt, use_cache=True)
    base_past = base_out.past_key_values
    with v(model):
        steer_out = model(prompt, use_cache=True)
    steer_past = steer_out.past_key_values

    chunks = [
        _kl_per_pos(
            torch.log_softmax(steer_out.logits[:, -1:, :].float(), dim=-1)[0],
            torch.log_softmax(base_out.logits[:, -1:, :].float(), dim=-1)[0],
        ).cpu()
    ]

    prev_tokens = gen[:-1]
    for start in range(0, prev_tokens.shape[0], chunk_size):
        chunk = prev_tokens[start:start + chunk_size].unsqueeze(0)
        base_out = model(chunk, past_key_values=base_past, use_cache=True)
        base_past = base_out.past_key_values
        with v(model):
            steer_out = model(chunk, past_key_values=steer_past, use_cache=True)
        steer_past = steer_out.past_key_values
        chunks.append(
            _kl_per_pos(
                torch.log_softmax(steer_out.logits.float(), dim=-1)[0],
                torch.log_softmax(base_out.logits.float(), dim=-1)[0],
            ).cpu()
        )

    return torch.cat(chunks)


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(torch.tensor(xs).quantile(q))


@torch.no_grad()
def measure_kl(
    v: Vector,
    model: nn.Module,
    tok,
    prompts=None,
    *,
    T: int = 50,
    do_sample: bool = False,
    device: str | torch.device = "cuda",
) -> dict:
    """Roll out under steering, score under base+steer. Returns scalar stats
    plus per-token arrays (mean, p95, max) of length T.
    """
    prompts = _tokenize(prompts, tok)
    all_kls = []
    per_t = [[] for _ in range(T)]
    # per_prompt_per_t[i][t] = KL of prompt i at gen-token t (NaN if not generated)
    per_prompt_per_t: list[list[float]] = []

    for idx, pids in enumerate(tqdm(prompts, desc="measure_kl", mininterval=60)):
        with v(model):
            gen = _generate(model, pids, T, tok, do_sample, device)
        n_gen = gen.shape[0]
        if n_gen == 0:
            continue
        full_ids = torch.cat([pids.to(device), gen])
        if idx == 0 and not _demo_logged["flag"]:
            _demo_logged["flag"] = True
            base_gen = _generate(model, pids, T, tok, do_sample, device)
            base_full = torch.cat([pids.to(device), base_gen])
            decoded_base = tok.decode(base_full, skip_special_tokens=False)
            decoded_steer = tok.decode(full_ids, skip_special_tokens=False)
            # SHOULD: BASE and STEER both coherent; STEER differs from BASE but does not collapse.
            # Truncate to keep iso-KL bracket logs scannable; full text in trajectory.json/debug_first.
            head = 400
            logger.info(
                f"SHOULD: c=0 vs c={v.cfg.coeff:+.4f} both coherent, steer differs but does not collapse.\n"
                f"=== CALIBRATE demo (T={T}, first {head} chars each) ===\n"
                f"-- BASE  : {decoded_base[:head]!r}\n"
                f"-- STEER : {decoded_steer[:head]!r}\n"
                f"=== /CALIBRATE ==="
            )
        kls = _kl_generated_incremental(v, model, pids, gen, device)
        all_kls.append(kls)
        row = [float("nan")] * T
        for i in range(n_gen):
            per_t[i].append(float(kls[i]))
            row[i] = float(kls[i])
        per_prompt_per_t.append(row)

    cat = torch.cat(all_kls)
    return {
        "kl_mean": float(cat.mean()),
        "kl_p50": float(cat.quantile(0.50)),
        "kl_p90": float(cat.quantile(0.90)),
        "kl_p95": float(cat.quantile(0.95)),
        "kl_max": float(cat.max()),
        "n_pos": int(cat.numel()),
        "per_t_mean": [sum(xs) / len(xs) if xs else 0.0 for xs in per_t],
        "per_t_p95": [_quantile(xs, 0.95) for xs in per_t],
        "per_t_max": [max(xs) if xs else 0.0 for xs in per_t],
        "per_prompt_per_t": per_prompt_per_t,
    }


def calibrate_iso_kl(
    v: Vector,
    model: nn.Module,
    tok,
    prompts=None,
    *,
    target_kl: float = 1.0,
    target_stat: str = "kl_p95",
    bracket: tuple[float, float] = (0.01, 4096.0),
    tol: float = 0.05,
    max_iters: int = 12,
    T: int = 50,
    device: str | torch.device = "cuda",
    sign: float = 1.0,
    sign_probe: Callable[[Vector], float] | None = None,
    sign_probe_c: float = 1.0,
) -> tuple[float, list[dict]]:
    """log-log Illinois bisection on `target_stat`. History keeps per-token
    arrays so we can plot the trajectory after."""
    _demo_logged["flag"] = False
    prompts = _tokenize(prompts, tok)
    history: list[dict] = []

    if sign_probe is not None:
        v.cfg.coeff = +sign_probe_c
        score_pos = sign_probe(v)
        v.cfg.coeff = -sign_probe_c
        score_neg = sign_probe(v)
        chosen = +1.0 if score_pos >= score_neg else -1.0
        logger.info(
            f"sign_probe: +c={sign_probe_c:+.2f} -> {score_pos:+.3f} | "
            f"-c={-sign_probe_c:+.2f} -> {score_neg:+.3f} | "
            f"chosen sign={chosen:+.0f}"
        )
        sign = sign * chosen

    def eval_at(c: float) -> float:
        v.cfg.coeff = sign * c
        m = measure_kl(v, model, tok, prompts, T=T, do_sample=False, device=device)
        history.append({"coeff": sign * c, "coeff_abs": c, "sign": sign, **m})
        logger.info(f"  c={sign * c:+.4f} mean={m['kl_mean']:.3f} "
                    f"p50={m['kl_p50']:.3f} p90={m['kl_p90']:.3f} "
                    f"p95={m['kl_p95']:.3f} max={m['kl_max']:.3f} n={m['n_pos']}")
        return m[target_stat]

    lo, hi = bracket
    log_target = math.log(target_kl)

    mid = (lo * hi) ** 0.5
    v_mid = eval_at(mid)
    if v_mid < target_kl:
        c_lo, v_lo = mid, v_mid
        c = mid
        c_hi, v_hi = hi, None
        while c < hi:
            c *= 2.0
            val = eval_at(c)
            if val >= target_kl:
                c_hi, v_hi = c, val
                break
            c_lo, v_lo = c, val
        else:
            raise ValueError(
                f"calibrate {v.cfg.method}: {target_stat}={v_lo:.4g} below target_kl={target_kl:.4g} "
                f"at max coeff={sign * c_lo:+.4g}; increase bracket high or lower target_kl"
            )
    else:
        c_hi, v_hi = mid, v_mid
        c = mid
        c_lo, v_lo = lo, None
        while c > lo:
            c /= 2.0
            val = eval_at(c)
            if val <= target_kl:
                c_lo, v_lo = c, val
                break
            c_hi, v_hi = c, val
        else:
            raise ValueError(
                f"calibrate {v.cfg.method}: {target_stat}={v_hi:.4g} above target_kl={target_kl:.4g} "
                f"at min coeff={sign * c_hi:+.4g}; decrease bracket low or raise target_kl"
            )

    stale_lo = stale_hi = 0
    for _ in tqdm(range(max_iters), desc=f"calib {v.cfg.method}", mininterval=60, leave=False):
        if v_lo is not None and v_hi is not None and v_lo > 0 and v_hi > 0:
            log_c_lo, log_c_hi = math.log(c_lo), math.log(c_hi)
            log_v_lo = math.log(v_lo) - (math.log(2) if stale_lo >= 2 else 0.0)
            log_v_hi = math.log(v_hi) - (math.log(2) if stale_hi >= 2 else 0.0)
            denom = log_v_hi - log_v_lo
            if abs(denom) < 1e-6:
                c_new = math.sqrt(c_lo * c_hi)
            else:
                t = (log_target - log_v_lo) / denom
                log_c_new = log_c_lo + t * (log_c_hi - log_c_lo)
                c_new = math.exp(log_c_new)
                if not (c_lo < c_new < c_hi):
                    c_new = math.sqrt(c_lo * c_hi)
        else:
            c_new = math.sqrt(c_lo * c_hi)

        v_new = eval_at(c_new)
        if abs(v_new - target_kl) < tol:
            return sign * c_new, history
        if v_new < target_kl:
            c_lo, v_lo = c_new, v_new
            stale_lo = 0
            stale_hi += 1
        else:
            c_hi, v_hi = c_new, v_new
            stale_hi = 0
            stale_lo += 1

    best = min(history, key=lambda h: abs(h[target_stat] - target_kl))
    return best["coeff"], history
