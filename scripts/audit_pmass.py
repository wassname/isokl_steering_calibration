"""Empirical audit of branch_pmass: load model, generate one rollout per alpha,
print decoded gen text and top-k tokens at the prefill end-point, and dump
everything to JSON for review.

Goal: distinguish between
  (a) pmass measurement is wrong (top tokens at prefill end DON'T match the
      pmass we read out of the schema-token groups), vs
  (b) pmass is right but the model just doesn't put mass on schema tokens
      naturally (steering isn't the problem, the prompt+prefill is), vs
  (c) pmass is right and steering really does collapse coherence.

For first prompt per alpha:
  - decoded full generation text
  - per fork point: top-10 tokens with prob, plus pmass(true)+pmass(false), p_true

Usage:
  uv run --extra all python scripts/audit_pmass.py \
      --model Qwen/Qwen3.5-0.8B --window 64 --out audit_pmass.json
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from loguru import logger

# Import the live module so any audit reflects current code
from iso_kl_figure import (
    MeanDiffC, PCAC, DirectionalAblationC,
    train, calibrate_iso_kl,
)
from iso_kl_figure.branch_pmass import collect_choice_token_ids, branch_pmass


# Re-import constants from run_cell so audit uses the same prompts/schema
import importlib.util, sys
_rc_path = Path(__file__).parent / "run_cell.py"
_spec = importlib.util.spec_from_file_location("_run_cell", _rc_path)
_rc = importlib.util.module_from_spec(_spec)
sys.modules["_run_cell"] = _rc
_spec.loader.exec_module(_rc)

CALIB_PROMPTS = _rc.CALIB_PROMPTS
EVAL_PROMPTS = _rc.EVAL_PROMPTS
_QUESTIONS = _rc._QUESTIONS
_SCHEMA = _rc._SCHEMA
PREFILL_STR = _rc.PREFILL_STR
POS_NEG = _rc.POS_NEG
METHOD_MAP = {"mean_diff": MeanDiffC, "pca": PCAC, "directional_ablation": DirectionalAblationC}


@dataclass
class Args:
    model: str = "Qwen/Qwen3.5-0.8B"
    method: str = "mean_diff"
    seed: int = 0
    window: int = 64
    layer_frac: float = 0.6
    target_kl: float = 1.0
    device: str = "cuda"
    dtype: str = "bfloat16"
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    fork_points: tuple[int, ...] = (0, 8, 16, 32, 64)
    out: str = "audit_pmass.json"
    top_k: int = 10
    use_qa_prompt: bool = True   # True: yes/no q+_SCHEMA; False: long-form EVAL_PROMPTS[0]
    skip_calib: bool = False     # if True, use fixed_coeffs as raw c per alpha (no iso-KL bisection)
    fixed_coeffs: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)  # used when skip_calib=True; aligned with --alphas
    cross_check_branch_pmass: bool = True  # also call branch_pmass and compare to local recompute


def _set_seed(s: int):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


@torch.no_grad()
def topk_at_prefill_end(v, model, tok, prompt_ids, rolled_ids, fork_points,
                        prefill_str, a_ids, b_ids, k=10, device="cuda"):
    """Mirror branch_pmass logic but ALSO return top-k tokens for inspection."""
    import copy
    pids = prompt_ids.to(device); rolled = rolled_ids.to(device)
    P = pids.shape[0]; T = rolled.shape[0]
    pre_t = torch.tensor(tok.encode(prefill_str, add_special_tokens=False),
                         device=device, dtype=torch.long)
    a_t = torch.tensor(list(a_ids), dtype=torch.long, device=device)
    b_t = torch.tensor(list(b_ids), dtype=torch.long, device=device)
    all_t = torch.cat([a_t, b_t])
    out_per_fork = []
    for t in fork_points:
        if t > T:
            out_per_fork.append({"t": int(t), "skipped": True, "reason": f"t>T={T}"})
            continue
        prefix = rolled[:t]
        seq = torch.cat([pids, prefix, pre_t]).unsqueeze(0)
        with v(model):
            logits = model(seq).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        # top-k
        tk_p, tk_i = probs.topk(k)
        topk = [(tok.decode([int(i)]), float(p)) for p, i in zip(tk_p.tolist(), tk_i.tolist())]
        pa = float(probs[a_t].sum()); pb = float(probs[b_t].sum())
        pm = pa + pb
        pt = pa / pm if pm > 0 else float("nan")
        out_per_fork.append({
            "t": int(t),
            "skipped": False,
            "topk": topk,
            "p_true_group": pa,
            "p_false_group": pb,
            "pmass": pm,
            "p_true": pt,
            "argmax": tok.decode([int(probs.argmax())]),
        })
    return out_per_fork


def main(a: Args):
    _set_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, a.dtype)
    logger.info(f"loading model={a.model}")
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer = int(a.layer_frac * n_layers)
    cfg_cls = METHOD_MAP[a.method]
    cfg = cfg_cls(coeff=1.0, layers=(layer,))

    # Train + calibrate
    pos = [tok.apply_chat_template([{"role": "user", "content": u},
                                    {"role": "assistant", "content": p}], tokenize=False)
           for u, (p, _) in zip(CALIB_PROMPTS, POS_NEG)]
    neg = [tok.apply_chat_template([{"role": "user", "content": u},
                                    {"role": "assistant", "content": n}], tokenize=False)
           for u, (_, n) in zip(CALIB_PROMPTS, POS_NEG)]
    v = train(model, tok, pos, neg, cfg, batch_size=4, max_length=128)
    if a.skip_calib:
        c_star = 1.0  # fixed_coeffs are absolute
        logger.info(f"skip_calib: using fixed_coeffs={a.fixed_coeffs} as raw c")
    else:
        c_star, _ = calibrate_iso_kl(v, model, tok, CALIB_PROMPTS,
                                     target_kl=a.target_kl, target_stat="kl_p95",
                                     T=a.window, device=a.device)
        logger.info(f"c_star={c_star:+.4f}")

    a_ids, b_ids = collect_choice_token_ids(tok)
    logger.info(f"a_ids (true-group)={a_ids} -> tokens={[tok.decode([i]) for i in a_ids]}")
    logger.info(f"b_ids (false-group)={b_ids} -> tokens={[tok.decode([i]) for i in b_ids]}")

    # Single prompt
    if a.use_qa_prompt:
        prompt = f"{_QUESTIONS[0]}\n\n{_SCHEMA}"
    else:
        prompt = EVAL_PROMPTS[0]
    logger.info(f"prompt: {prompt[:120]}...")

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt").input_ids[0]

    out = {
        "model": a.model, "method": a.method, "seed": a.seed,
        "c_star": c_star, "layer": layer,
        "prompt": prompt, "use_qa_prompt": a.use_qa_prompt,
        "prefill": PREFILL_STR,
        "a_ids": a_ids, "b_ids": b_ids,
        "a_tokens": [tok.decode([i]) for i in a_ids],
        "b_tokens": [tok.decode([i]) for i in b_ids],
        "fork_points": list(a.fork_points),
        "alphas": {},
    }

    for i_alpha, alpha in enumerate(a.alphas):
        if a.skip_calib:
            v.cfg.coeff = a.fixed_coeffs[i_alpha]
        else:
            v.cfg.coeff = alpha * c_star
        logger.info(f"=== alpha={alpha} coeff={v.cfg.coeff:+.4f} ===")
        with v(model):
            gen_out = model.generate(
                ids.unsqueeze(0).to(a.device),
                max_new_tokens=a.window,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                do_sample=False,
                return_dict_in_generate=True,
            )
        gen = gen_out.sequences[0, ids.shape[0]:]
        gen_text = tok.decode(gen, skip_special_tokens=False)
        gen_len = int(gen.shape[0])
        logger.info(f"  gen_len={gen_len} text[:200]={gen_text[:200]!r}")

        per_fork = topk_at_prefill_end(
            v, model, tok, ids, gen, list(a.fork_points),
            PREFILL_STR, a_ids, b_ids, k=a.top_k, device=a.device,
        )
        # cross-check: compare to production branch_pmass output
        bp_compare = None
        if a.cross_check_branch_pmass:
            bp = branch_pmass(
                v, model, tok, ids, gen, list(a.fork_points),
                PREFILL_STR, a_ids, b_ids,
                rollout_cache=getattr(gen_out, "past_key_values", None),
                device=a.device,
            )
            bp_compare = {
                "pmass": bp["pmass"], "p_true": bp["p_true"],
                "argmax_str": bp["argmax_str"], "was_thinking": bp["was_thinking"],
            }
            for row, bpm, bpt, bam in zip(per_fork, bp["pmass"], bp["p_true"], bp["argmax_str"]):
                if row.get("skipped"): continue
                local_pm = row["pmass"]; local_pt = row["p_true"]; local_am = row["argmax"]
                tag = "OK"
                if not (abs(local_pm - bpm) < 1e-3 and (local_am == bam)):
                    tag = "MISMATCH"
                logger.info(f"    cross-check t={row['t']:>3} {tag}: local pm={local_pm:.4f} argmax={local_am!r} | branch_pmass pm={bpm:.4f} argmax={bam!r}")
        # log inline
        for row in per_fork:
            if row.get("skipped"):
                logger.info(f"  t={row['t']:>3} SKIPPED ({row['reason']})")
            else:
                top3 = ", ".join(f"{tok!r}={p:.3f}" for tok, p in row["topk"][:3])
                logger.info(f"  t={row['t']:>3} pmass={row['pmass']:.3f} "
                            f"p_true={row['p_true']:.3f} argmax={row['argmax']!r}  top3=[{top3}]")
        out["alphas"][str(alpha)] = {
            "coeff": float(v.cfg.coeff),
            "gen_text": gen_text, "gen_len": gen_len, "per_fork": per_fork,
            "branch_pmass_compare": bp_compare,
        }

    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"DONE -> {a.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
