"""End-to-end runner for one (model, method, seed, window) cell.

Artefacts (per cell, under outputs/<run_id>/):
  calib.json       c_star + run metadata (model, method, seed, layer, window).
  history.json     bisection history (without per-token KL arrays).
  trajectory.json  per-token KL on EVAL_PROMPTS for each alpha:
                     per_t_p95_kl[alpha]:        list[T]
                     per_prompt_per_t_kl[alpha]: list[N_prompts][T]
  pmass.json       forked-answer probability mass at fork_points:
                     pmass[alpha]:      yes/no reasoning prompts (legacy probe)
                     pmass_eval[alpha]: SAME prompts as trajectory.json (paired w KL)
                     gen_lens_qa[alpha], gen_lens_eval[alpha]: T per rollout
                       (use for right-censoring -- NaN at t > T means rollout
                       EOS'd before that fork, NOT a measurement failure).
                     debug_first[alpha]: gen_text + per-fork pmass for the
                       FIRST qa & eval prompt (sanity-check; see survival.py).
                   pmass_eval is the metric to use when you want a per-trajectory
                   coherence signal aligned with the KL trajectory.
  results.csv      one row per alpha with kl_p95/mean/max.

Flow:
 1. Load model + tokenizer; set seed.
 2. Build pos/neg pair (cheap content-vs-refusal); train Vector v.
 3. Calibrate iso-KL: bisect coeff so per-token kl_p95 hits target_kl (default 1)
    over T=window tokens on CALIB_PROMPTS.
 4. For each alpha, set coeff=alpha*c_star and:
     a) measure_kl on EVAL_PROMPTS -> per_prompt_per_t_kl trajectories.
     b) (--compute-pmass) yes/no questions: rollout, then at each fork point
        prefill `\n{"choice": ` and read mass on {true/True/ true/.../1/ 1}
        and {false/.../0}. Skip with --skip-pmass-qa.
     c) (--compute-pmass) repeat (b) on EVAL_PROMPTS -> pmass_eval, paired
        with the KL trajectory on the same rollouts. This is the honest
        coherence signal; KL is largely a calibration tautology at alpha=1.

Fork-point modes:
  --fork-step N    linear: {0, N, 2N, ..., window}
  --fork-log       log-spaced: {0, 1, 2, 4, 8, ..., window} (with --n-log-forks)

KV-cache:
  Rollouts use generate(..., return_dict_in_generate=True). branch_pmass
  uses full-recompute by default (cache reuse adds ~10% bf16 noise; verified
  by scripts/audit_pmass.py). Pass use_rollout_cache=True only if you accept
  the noise.

Usage (tyro):
  python scripts/run_cell.py --model Qwen/Qwen2.5-0.5B-Instruct --method mean_diff \
      --seed 0 --window 512 --compute-pmass --skip-pmass-qa --fork-log
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from loguru import logger

from iso_kl_figure import (
    SteeringConfig, MeanDiffC, PCAC, DirectionalAblationC,
    train, calibrate_iso_kl, measure_kl,
)
from iso_kl_figure.branch_pmass import branch_pmass, collect_choice_token_ids
from iso_kl_figure.calibrate import _eos_token_ids
from iso_kl_figure.target import _get_blocks


CALIB_PROMPTS = [
    "The eiffel tower is in Paris.",
    "Tell me a greentext story about a small village during the Martian carrot bubble.",
    "Walk me through computing the integral of x^2 from 0 to 1.",
    "Briefly describe how plate tectonics shape mountain ranges.",
]

# EVAL prompts are yes/no questions paired with schema -- same contract as
# tinymfv/guided.py so branch_pmass at any fork can score format-compliance.
# The model is steered into <think>\n at the prompt boundary; at fork t we
# inject "\nI should answer now.</think>\n{\"choice\": " and read p(true|false).
EVAL_PROMPTS = [
    "Is the sky blue primarily because of Rayleigh scattering of sunlight?",
    "Does Hamlet die at the end of Shakespeare's play 'Hamlet'?",
    "Is excessive money supply growth one cause of inflation in modern economies?",
    "Do octopuses have three hearts and blue copper-based blood?",
    "Does photosynthesis use light energy to convert CO2 and water into glucose?",
    "Was the Eiffel Tower originally built as the entrance arch for the 1889 World's Fair?",
    "Can plate tectonic collisions form mountain ranges over geological time?",
    "Is a circular import one possible cause of a Python script crashing on import?",
    "Does water boil at a lower temperature at higher altitudes due to reduced atmospheric pressure?",
    "Is the mitochondrion commonly described as the powerhouse of the eukaryotic cell?",
    "Did the Apollo 11 mission land humans on the Moon in July 1969?",
    "Is Mount Everest the tallest mountain on Earth measured from base to peak?",
    "Can a closure in JavaScript capture variables from its enclosing lexical scope?",
    "Does the human body have exactly 206 bones in adulthood?",
    "Is sound generally faster in solids than in air at the same temperature?",
    "Was the Great Wall of China built primarily as a single continuous wall in one dynasty?",
    "Does relativity predict that time dilates for an observer moving at relativistic speeds?",
    "Is DNA replication semi-conservative, with each daughter molecule keeping one parent strand?",
    "Did the Roman Empire formally split into Western and Eastern halves in the 4th century CE?",
    "Is gradient descent guaranteed to find the global minimum of an arbitrary non-convex loss?",
    "Does the moon always show the same face to Earth because of tidal locking?",
    "Was Shakespeare a contemporary of Queen Elizabeth I of England?",
    "Is the Pacific Ocean larger in surface area than the Atlantic Ocean?",
    "Does a transformer architecture rely on self-attention rather than recurrence for sequence modeling?",
]

# pmass diagnostic prompts share the same schema as EVAL above.
_SCHEMA = (
    'Reason for at least 80 words before the final answer. Then answer only with '
    '{"choice": true} or {"choice": false}. If you use numeric JSON, use 1 for true and 0 for false.'
)
_QUESTIONS = [
    "Is the Eiffel Tower located in Paris, France?",
    "Do dolphins breathe air through lungs rather than gills?",
    "Was Albert Einstein born in Germany?",
    "Can humans naturally see ultraviolet light?",
    "Did the Roman Empire fall in 476 CE?",
    "Are tomatoes botanically classified as fruits?",
    "Is the speed of light approximately 300,000 km/s?",
    "Do octopuses have three hearts?",
]
PREFILL_STR = '\n{"choice": '

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
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    fork_step: int = 5
    fork_log: bool = False        # log-spaced fork points {0,1,2,4,8,...,window}
    n_log_forks: int = 14         # number of log-spaced forks (incl. 0)
    compute_pmass: bool = False
    skip_pmass_qa: bool = False   # skip yes/no pmass loop, only do paired pmass_eval
    render_figs: bool = False     # render single-run survival + spaghetti + KL pngs
    render_threshold: float = 0.95


def _set_seed(s: int):
    import random
    import numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _render_chat(tok, msgs: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if tok.chat_template is None:
        rendered = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                rendered.append(f"User: {content}\n")
            elif role == "assistant":
                rendered.append(f"Assistant: {content}")
            else:
                raise ValueError(f"plain prompt renderer does not support role={role!r}")
        if add_generation_prompt:
            rendered.append("Assistant: ")
        return "".join(rendered)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def _build_guided_prompt(tok, user_text: str, schema_hint: str = _SCHEMA) -> str:
    """Match tinymfv/guided.py: chat template + '<think>\\n' suffix so the model
    is in thinking mode at every fork point. branch_pmass detects this and
    splices '\\nI should answer now.</think>{prefill}' before scoring."""
    full_user = f"{user_text}\n\n{schema_hint}" if schema_hint else user_text
    msgs = [{"role": "user", "content": full_user}]
    p = _render_chat(tok, msgs, add_generation_prompt=True)
    return p + "<think>\n"


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
    model_kwargs = {"torch_dtype": dtype}
    if "bnb-4bit" in a.model.lower():
        model_kwargs["device_map"] = {"": a.device}
        model = AutoModelForCausalLM.from_pretrained(a.model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, **model_kwargs).to(a.device)
    model.eval()

    n_layers = len(_get_blocks(model))
    layer = int(a.layer_frac * n_layers)
    logger.info(f"model={a.model} n_layers={n_layers} target_layer={layer}")

    cfg_cls = METHOD_MAP[a.method]
    cfg = cfg_cls(coeff=1.0, layers=(layer,))

    pos = [_render_chat(tok, [{"role": "user", "content": u},
                              {"role": "assistant", "content": p}], add_generation_prompt=False)
           for u, (p, _) in zip(CALIB_PROMPTS, POS_NEG)]
    neg = [_render_chat(tok, [{"role": "user", "content": u},
                              {"role": "assistant", "content": n}], add_generation_prompt=False)
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
    # Strip per_prompt_per_t from history to keep file size small.
    hist_slim = [{k: v_ for k, v_ in h.items() if k != "per_prompt_per_t"}
                 for h in history]
    (out_dir / "history.json").write_text(json.dumps(hist_slim, indent=2))
    (out_dir / "calib.json").write_text(json.dumps({
        "c_star": c_star, "target_kl": a.target_kl, "window": a.window,
        "method": a.method, "model": a.model, "seed": a.seed, "layer": layer,
    }, indent=2))

    # Sanity-check pmass on the BASE model (no steering): should be ~1.0,
    # otherwise the prefill/schema isn't priming the right tokens.
    a_ids, b_ids = collect_choice_token_ids(tok)
    if a.compute_pmass:
        logger.info(f"choice ids: a(true)={a_ids} b(false)={b_ids}")

    # -- trajectory + pmass at each alpha on held-out prompts
    rows = []
    if a.fork_log:
        # log-spaced including 0 and window: {0, 1, 2, 4, 8, ..., window}
        import numpy as _np
        raw = _np.unique(_np.round(_np.geomspace(1, a.window, a.n_log_forks - 1)).astype(int))
        fork_points = [0] + [int(x) for x in raw]
        fork_points = sorted(set(fp for fp in fork_points if fp <= a.window))
    else:
        fork_points = list(range(0, a.window + 1, a.fork_step))
    logger.info(f"fork_points (n={len(fork_points)}): {fork_points}")
    trajectory: dict[str, list] = {}
    per_prompt_traj: dict[str, list] = {}
    pmass_all: dict[str, list] = {}
    pmass_eval_all: dict[str, list] = {}   # pmass paired with EVAL_PROMPTS (same prompts as KL)
    p_true_all: dict[str, list] = {}
    argmax_all: dict[str, list] = {}
    thinking_all: dict[str, list] = {}
    answer_label_all: dict[str, list] = {}
    gen_lens_qa: dict[str, list] = {}      # T per (alpha, qa-prompt) -- right-censoring info
    gen_lens_eval: dict[str, list] = {}    # T per (alpha, eval-prompt)
    debug_first: dict[str, dict] = {}      # first prompt per alpha: gen_text + top-5 at each fork

    def write_eval_outputs() -> None:
        (out_dir / "trajectory.json").write_text(json.dumps({
            "fork_points_full": list(range(a.window)),
            "per_t_p95_kl": trajectory,
            "per_prompt_per_t_kl": per_prompt_traj,
        }, indent=2))
        (out_dir / "pmass.json").write_text(json.dumps({
            "fork_points": fork_points,
            "pmass": pmass_all,
            "pmass_eval": pmass_eval_all,
            "p_true": p_true_all,
            "argmax_str": argmax_all,
            "was_thinking": thinking_all,
            "answer_label": answer_label_all,
            "gen_lens_qa": gen_lens_qa,
            "gen_lens_eval": gen_lens_eval,
            "debug_first": debug_first,
            "prefill": PREFILL_STR,
            "schema": _SCHEMA,
            "questions": _QUESTIONS,
            "computed": a.compute_pmass,
            "completed_alphas": list(trajectory.keys()),
        }, indent=2))
        import csv
        with open(out_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["alpha", "coeff", "kl_p95", "kl_mean", "kl_max"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Pre-build guided EVAL ids ONCE so measure_kl and pmass_eval roll out the
    # same prompt -- otherwise spaghetti coloring (KL trajectory colored by
    # pmass) is meaningless because the rollouts diverge.
    eval_prompt_strs = [_build_guided_prompt(tok, p, _SCHEMA) for p in EVAL_PROMPTS]
    eval_ids_list = [
        tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0]
        for s in eval_prompt_strs
    ]
    for alpha in a.alphas:
        v.cfg.coeff = alpha * c_star
        logger.info(f"=== eval alpha={alpha} c={v.cfg.coeff:+.4f} ===")
        m = measure_kl(v, model, tok, eval_ids_list, T=a.window, device=a.device)
        trajectory[str(alpha)] = m["per_t_p95"]
        per_prompt_traj[str(alpha)] = m["per_prompt_per_t"]
        rows.append({"alpha": alpha, "coeff": v.cfg.coeff, "kl_p95": m["kl_p95"],
                     "kl_mean": m["kl_mean"], "kl_max": m["kl_max"]})

        # pmass per held-out prompt
        pm_for_alpha, pt_for_alpha, ax_for_alpha, wt_for_alpha, ans_for_alpha = [], [], [], [], []
        gen_lens_qa_alpha = []
        gen_lens_eval_alpha = []
        if a.compute_pmass and not a.skip_pmass_qa:
            for q_idx, q in enumerate(_QUESTIONS):
                prompt_str = _build_guided_prompt(tok, q, _SCHEMA)
                ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids[0]
                pad = tok.pad_token_id
                with v(model):
                    gen_out = model.generate(
                        ids.unsqueeze(0).to(a.device),
                        max_new_tokens=a.window,
                        pad_token_id=pad, eos_token_id=_eos_token_ids(tok),
                        do_sample=False,
                        use_cache=True,
                        return_dict_in_generate=True,
                    )
                gen = gen_out.sequences[0, ids.shape[0]:]
                gen_lens_qa_alpha.append(int(gen.shape[0]))
                # KV cache from steered rollout: ~100x speedup at each fork.
                pm = branch_pmass(
                    v, model, tok, ids, gen, fork_points,
                    PREFILL_STR, a_ids, b_ids,
                    rollout_cache=gen_out.past_key_values,
                    device=a.device,
                )
                pm_for_alpha.append(pm["pmass"])
                pt_for_alpha.append(pm["p_true"])
                ax_for_alpha.append(pm["argmax_str"])
                wt_for_alpha.append(pm["was_thinking"])
                ans_for_alpha.append([
                    "true" if s in {"true", "True", " true", " True", "1", " 1"}
                    else "false" if s in {"false", "False", " false", " False", "0", " 0"}
                    else "other"
                    for s in pm["argmax_str"]
                ])
                # debug dump for first QA prompt only
                if q_idx == 0:
                    gen_text = tok.decode(gen, skip_special_tokens=False)
                    debug_first.setdefault(str(alpha), {})["qa"] = {
                        "prompt": prompt_str, "gen_text": gen_text, "gen_len": int(gen.shape[0]),
                        "pmass_per_fork": pm["pmass"], "p_true_per_fork": pm["p_true"],
                        "argmax_per_fork": pm["argmax_str"],
                    }
                    logger.info(f"  [debug] alpha={alpha} qa[0] gen_len={gen.shape[0]} text[:120]={gen_text[:120]!r}")
                    for t, pmv, ptv, am in zip(fork_points, pm["pmass"], pm["p_true"], pm["argmax_str"]):
                        logger.info(f"    t={t:>3} pmass={pmv:.3f} p_true={ptv:.3f} argmax={am!r}")
        pmass_all[str(alpha)] = pm_for_alpha
        p_true_all[str(alpha)] = pt_for_alpha
        argmax_all[str(alpha)] = ax_for_alpha
        thinking_all[str(alpha)] = wt_for_alpha
        answer_label_all[str(alpha)] = ans_for_alpha
        gen_lens_qa[str(alpha)] = gen_lens_qa_alpha

        # Paired pmass on EVAL_PROMPTS (same long-form prompts as KL) so we can
        # color KL trajectories by pmass at each fork. Long-form prompts won't
        # naturally produce schema tokens, but the prefill forces the question
        # "if you committed now, can you still produce a valid choice?" -- which
        # is exactly the coherence signal we want.
        pm_eval_for_alpha = []
        if a.compute_pmass:
            for p_idx, p in enumerate(EVAL_PROMPTS):
                prompt_str = eval_prompt_strs[p_idx]
                ids = eval_ids_list[p_idx]
                pad = tok.pad_token_id
                with v(model):
                    gen_out = model.generate(
                        ids.unsqueeze(0).to(a.device),
                        max_new_tokens=a.window,
                        pad_token_id=pad, eos_token_id=_eos_token_ids(tok),
                        do_sample=False,
                        use_cache=True,
                        return_dict_in_generate=True,
                    )
                gen = gen_out.sequences[0, ids.shape[0]:]
                gen_lens_eval_alpha.append(int(gen.shape[0]))
                pm = branch_pmass(
                    v, model, tok, ids, gen, fork_points,
                    PREFILL_STR, a_ids, b_ids,
                    rollout_cache=gen_out.past_key_values,
                    device=a.device,
                )
                pm_eval_for_alpha.append(pm["pmass"])
                if p_idx == 0:
                    gen_text = tok.decode(gen, skip_special_tokens=False)
                    debug_first.setdefault(str(alpha), {})["eval"] = {
                        "prompt": prompt_str, "gen_text": gen_text, "gen_len": int(gen.shape[0]),
                        "pmass_per_fork": pm["pmass"], "p_true_per_fork": pm["p_true"],
                        "argmax_per_fork": pm["argmax_str"],
                    }
                    logger.info(f"  [debug] alpha={alpha} eval[0] gen_len={gen.shape[0]} text[:120]={gen_text[:120]!r}")
                    for t, pmv, ptv, am in zip(fork_points, pm["pmass"], pm["p_true"], pm["argmax_str"]):
                        logger.info(f"    t={t:>3} pmass={pmv:.3f} p_true={ptv:.3f} argmax={am!r}")
        pmass_eval_all[str(alpha)] = pm_eval_for_alpha
        gen_lens_eval[str(alpha)] = gen_lens_eval_alpha
        # SHOULD: at alpha=1, mean(pmass at t=0) > 0.5 (model still respects schema).
        # ELSE: prefill string broken or chat template off.
        if a.compute_pmass:
            import numpy as _np
            pmass_rows = pm_eval_for_alpha if a.skip_pmass_qa else pm_for_alpha
            t0 = _np.array([row[0] for row in pmass_rows])
            logger.info(f"  alpha={alpha} pmass@t=0: mean={t0.mean():.3f} min={t0.min():.3f} max={t0.max():.3f}")
        write_eval_outputs()

    write_eval_outputs()
    if a.render_figs:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            "scripts/render_run_figs.py",
            "--run-dir", str(out_dir),
            "--threshold", str(a.render_threshold),
            "--model-contains", a.model.split("/")[-1],
        ]
        logger.info("rendering single-run figures")
        subprocess.run(cmd, check=True, cwd=repo_root)
    logger.info(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
