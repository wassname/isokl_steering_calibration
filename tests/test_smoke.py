"""Smoke test: train + calibrate + branch_pmass on a tiny random model.

Pass = runtime sanity. Distinguishing checks:
  - measure_kl returns kl > 0 at coeff > 0 (steer DID change distribution).
  - measure_kl returns kl ~= 0 at coeff = 0 (silent failure detector: if hooks
    leak, base==steer KL would be nonzero).
  - branch_pmass is in [0, 1].
  - branch_pmass changes between coeff=0 and coeff=large (sneaky-fail catch:
    if pmass is just identity-pass-through it would be invariant).
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iso_kl_figure import (
    SteeringConfig, MeanDiffC, train, measure_kl, attach, detach,
)
from iso_kl_figure.branch_pmass import (
    branch_pmass,
    build_chat_interrupt_suffix,
    collect_choice_token_ids,
)


MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_pipeline_smoke():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    cfg = MeanDiffC(coeff=1.0, layers=(n_layers // 2,))

    pos = ["Sure: <answer>", "Yes: <answer>", "Of course: <answer>", "Here: <answer>"]
    neg = ["No way.", "I refuse.", "Cannot help.", "Decline."]
    v = train(model, tok, pos, neg, cfg, batch_size=2, max_length=32)

    # KL must be ~0 at coeff=0 (no leak), and > 0 at large coeff
    v.cfg.coeff = 0.0
    m0 = measure_kl(v, model, tok, ["hello world"], T=4, device="cpu")
    assert m0["kl_p95"] < 1e-3, f"coeff=0 should give zero KL, got {m0['kl_p95']}"

    v.cfg.coeff = 5.0
    m1 = measure_kl(v, model, tok, ["hello world"], T=4, device="cpu")
    assert m1["kl_p95"] > 0.0, "coeff>0 should give nonzero KL"

    # per_t arrays length matches T
    assert len(m1["per_t_p95"]) == 4
    assert len(m1["per_t_max"]) == 4

    # branch_pmass: in [0, 1] and varies with coeff
    pids = tok("hello", return_tensors="pt").input_ids[0]
    rolled = pids[-2:].clone()
    prefill = ' {"choice": '
    fork = [0, 1, 2]
    a_ids, b_ids = collect_choice_token_ids(tok)
    assert a_ids and b_ids, "choice ids empty"

    v.cfg.coeff = 0.0
    p_zero = branch_pmass(v, model, tok, pids, rolled, fork, prefill,
                          a_ids, b_ids, device="cpu")
    v.cfg.coeff = 5.0
    p_steer = branch_pmass(v, model, tok, pids, rolled, fork, prefill,
                           a_ids, b_ids, device="cpu")
    tok.chat_template = (
        "{% for message in messages %}"
        "<|{{ message['role'] }}|>\n{{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    interrupt_ids = build_chat_interrupt_suffix(tok, "Final answer?", '{"Answer": ')
    assert tok.decode(interrupt_ids).endswith('{"Answer": ')
    p_interrupt = branch_pmass(
        v, model, tok, pids, rolled, fork, prefill,
        a_ids, b_ids, interrupt_suffix_ids=interrupt_ids, device="cpu",
    )
    for x in p_zero["pmass"] + p_steer["pmass"]:
        assert 0.0 <= x <= 1.0, f"pmass out of [0,1]: {x}"
    for x in p_interrupt["pmass"]:
        assert 0.0 <= x <= 1.0, f"chat-interrupt pmass out of [0,1]: {x}"
    diff = max(abs(a - b) for a, b in zip(p_zero["pmass"], p_steer["pmass"]))
    assert diff > 1e-8, "pmass invariant to coeff -- hook is dead"
    # p_true should be in [0,1] or NaN
    for pt in p_zero["p_true"] + p_steer["p_true"]:
        assert (pt != pt) or (0.0 <= pt <= 1.0), f"p_true out of [0,1]: {pt}"
