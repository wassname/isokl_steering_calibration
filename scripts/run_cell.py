"""End-to-end runner for one (model, method, seed, window) cell.

Flow:
1. Load model + tokenizer (HF), set seed.
2. Build pos/neg prompts (cheap pair); train the steering Vector v.
3. Calibrate iso-KL at target_kl=1 over T=window tokens. Save full history
   (incl. per-token KL arrays) to outputs/<run_id>/history.json.
4. Re-run measure_kl at coeff=alpha*c_star (alpha in {1, 2}) on a held-out
   prompt set so the trajectory plot reflects generalisation, not the
   calibration set itself. Save per-token p95 KL to trajectory.json.
5. For each held-out prompt, rollout T_eval tokens under the steered model,
   then branch-pmass at fork_points {0, 5, ..., T_eval}. Save to
   pmass.json. Use a JSON-format suffix so target tokens are well-defined.

Outputs one CSV row per (alpha, prompt) into outputs/<run_id>/results.csv
plus the artefacts above.

Usage (tyro):
  python scripts/run_cell.py --model Qwen/Qwen2.5-0.5B-Instruct --method mean_diff \
      --seed 0 --window 50 --run_id qwen05b_md_s0_w50
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from loguru import logger

from iso_kl_figure import (
    SteeringConfig, MeanDiffC, PCAC, DirectionalAblationC,
    train, calibrate_iso_kl, measure_kl,
)
from iso_kl_figure.branch_pmass import branch_pmass


CALIB_PROMPTS = [
    "The eiffel tower is in Paris.",
    "Tell me a greentext story about a small village during the Martian carrot bubble.",
    "Walk me through computing the integral of x^2 from 0 to 1.",
    "Briefly describe how plate tectonics shape mountain ranges.",
]

# Held-out prompts -- different from calibration; tests generalisation.
EVAL_PROMPTS = [
    "Explain why the sky is blue in two short sentences.",
    "Summarise the plot of Hamlet in one paragraph.",
    "List three causes of inflation.",
    "Write a haiku about an octopus librarian.",
]

# Pos/neg pair: a generic content-vs-refusal direction. Cheap, not the main
# point -- the figure tests calibration *behaviour*, not direction quality.
POS_NEG = [
    ("Sure, here is the answer:", "I cannot help with that."),
    ("Here are some details:", "Sorry, I can't assist with that."),
    ("Of course, let me explain.", "I won't be able to help."),
    ("Yes, that makes sense.", "No, I have to decline."),
]


METHOD_MAP = {
    "mean_diff": MeanDiffC,
    "pca": PCAC,
    "directional_ablation": DirectionalAblationC,
}


@dataclass
class Args:
    model: str
    method: str
    seed: int = 0
    window: int = 50
    run_id: str = ""
    layer_frac: float = 0.6
    target_kl: float = 1.0
    out_root: str = "outputs"
    device: str = "cuda"
    dtype: str = "bfloat16"
    suffix_str: str = ' Final answer in JSON: {"value": '
    target_words: list[str] = field(default_factory=lambda: ["true", "false", "yes", "no"])
    fork_step: int = 5


def _set_seed(s: int):
    import random
    import numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main(a: Args):
    if not a.run_id:
        a.run_id = f"{a.model.split('/')[-1]}_{a.method}_s{a.seed}_w{a.window}"
    out_dir = Path(a.out_root) / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "run.log", level="INFO")

    _set_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer = int(a.layer_frac * n_layers)
    logger.info(f"model={a.model} n_layers={n_layers} target_layer={layer}")

    cfg_cls = METHOD_MAP[a.method]
    cfg = cfg_cls(coeff=1.0, layers=(layer,))

    pos = [tok.apply_chat_template([{"role": "user", "content": u},
                                    {"role": "assistant", "content": p}],
                                   tokenize=False)
           for u, (p, _) in zip(CALIB_PROMPTS, POS_NEG)]
    neg = [tok.apply_chat_template([{"role": "user", "content": u},
                                    {"role": "assistant", "content": n}],
                                   tokenize=False)
           for u, (_, n) in zip(CALIB_PROMPTS, POS_NEG)]
    v = train(model, tok, pos, neg, cfg, batch_size=4, max_length=128)

    logger.info("=== calibrate ===")
    c_star, history = calibrate_iso_kl(
        v, model, tok, CALIB_PROMPTS,
        target_kl=a.target_kl, target_stat="kl_p95",
        T=a.window, device=a.device,
    )
    v.cfg.coeff = c_star
    logger.info(f"c_star = {c_star:+.4f}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "calib.json").write_text(json.dumps({
        "c_star": c_star, "target_kl": a.target_kl, "window": a.window,
        "method": a.method, "model": a.model, "seed": a.seed, "layer": layer,
    }, indent=2))

    # -- trajectory + pmass at alpha in {1, 2} on held-out prompts
    rows = []
    fork_points = list(range(0, a.window + 1, a.fork_step))
    trajectory: dict[str, list] = {}
    pmass_all: dict[str, list] = {}
    for alpha in (1.0, 2.0):
        v.cfg.coeff = alpha * c_star
        logger.info(f"=== eval alpha={alpha} c={v.cfg.coeff:+.4f} ===")
        m = measure_kl(v, model, tok, EVAL_PROMPTS, T=a.window, device=a.device)
        trajectory[str(alpha)] = m["per_t_p95"]
        rows.append({"alpha": alpha, "coeff": v.cfg.coeff, "kl_p95": m["kl_p95"],
                     "kl_mean": m["kl_mean"], "kl_max": m["kl_max"]})

        # pmass per held-out prompt
        pm_for_alpha = []
        for p in EVAL_PROMPTS:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt",
            ).input_ids[0]
            pad = tok.pad_token_id
            with v(model):
                gen = model.generate(
                    ids.unsqueeze(0).to(a.device),
                    max_new_tokens=a.window,
                    pad_token_id=pad, eos_token_id=tok.eos_token_id,
                    do_sample=False,
                )[0, ids.shape[0]:]
            pm = branch_pmass(
                v, model, tok, ids, gen, fork_points,
                a.suffix_str, a.target_words, device=a.device,
            )
            pm_for_alpha.append(pm["pmass"])
        pmass_all[str(alpha)] = pm_for_alpha

    (out_dir / "trajectory.json").write_text(json.dumps({
        "fork_points_full": list(range(a.window)),
        "per_t_p95_kl": trajectory,
    }, indent=2))
    (out_dir / "pmass.json").write_text(json.dumps({
        "fork_points": fork_points,
        "pmass": pmass_all,
        "suffix": a.suffix_str,
        "target_words": a.target_words,
    }, indent=2))
    import csv
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "coeff", "kl_p95", "kl_mean", "kl_max"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
