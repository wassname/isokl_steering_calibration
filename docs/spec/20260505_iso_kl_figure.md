# iso-kl-figure: spec

## Goal

Produce one figure (Figure 1) and one table (Table 1) that empirically support three claims: iso-KL calibration converges and generalizes (C1), the calibrated coefficient is not too cold (C2), and not too hot (C3). Show stability across 3 models x 3 seeds x 2 calibration windows.

## Scope

In:
- Port `measure_kl`, `calibrate_iso_kl`, minimal Vector/attach/config/target/extract from steering-lite.
- 3 methods: `mean_diff`, `directional_ablation`, `pca`.
- New `branch_pmass` metric: fork-and-teacher-force probability mass on a forced format answer token.
- Scripts producing TSV/CSV; plot and table modules consuming the CSVs.

Out:
- LessWrong post or paper draft.
- Method zoo beyond 3 methods.
- Threshold sweep, calibration-set-size sweep, norm-matching baseline.
- tinymfv integration.

## Requirements

- R1 (C1, calibration converges and generalizes): for every (method, model, seed, window), bisection terminates with calibration p95 within tolerance of 1.0; on a held-out prompt set p95 lands within [0.7, 1.4]. VERIFY: TSV row has converged=true and holdout_p95 in band; sneaky failure (overfits calibration prompts) caught by held-out column.
- R2 (C2, not too cold): target-axis Delta logit at calibrated alpha excludes 0 with 95% CI for each method, on each model. VERIFY: Table 1 row reports CI; sneaky failure (alpha approx 0) caught by alpha column in same row.
- R3 (C3, not too hot, NLL): base-NLL of full 50-token held-out generations stays within 2x of base at calibrated alpha; exceeds 4x of base at 2x calibrated alpha for at least one method per model. VERIFY: Table 1 base_nll_delta column.
- R4 (C3, not too hot, branch-pmass): mean branch-pmass-of-valid-answer at fork points {0, 5, ..., 50} stays within 0.1 of base pmass at calibrated alpha; drops by more than 0.3 at 2x alpha for at least one method per model. VERIFY: Table 1 branch_pmass column and Figure 1 lower subplot.
- R5 (sanity probe at 2x): max p95 KL at 2x alpha exceeds 1 nat in at least 2 of 3 methods on at least 2 of 3 models within 50 tokens. VERIFY: Figure 1 top subplot, alpha=2 panels show lines crossing reference.
- R6 (stability): seed band and window-style overlay in Figure 1 do not change the qualitative C1 conclusion. VERIFY: variance band visually narrow at alpha=1.

## Tasks

- [/] T1 (R*): scaffold repo (pyproject, justfile, README, AGENTS, spec).
  - verify: `just --list` lists recipes; `uv sync --extra all` resolves.
- [ ] T2 (R1, R2, R3, R4): port core code from steering-lite (calibrate, vector, attach, config, target, extract, 3 variants).
  - verify: imports clean; smoke test runs all 3 methods.
- [ ] T3 (R1, R6): extend calibrate history to save per-token KL arrays (`per_t_p95`, `per_t_max`).
  - verify: history dict contains per-token arrays of length T.
- [ ] T4 (R4): implement `branch_pmass` (fork at token t, append fixed format suffix, teacher-force one forward, sum p over `true`/`false` tokens).
  - verify: pmass in [0, 1]; pmass at base != pmass at coeff=large (sneaky-fail catch).
- [ ] T5 (R1..R5): implement `run_calibrate.py`, `run_trajectory.py`, `run_table.py`.
  - verify: CSVs created with expected columns and at least one row each on smoke.
- [ ] T6 (R*): implement `plot.py`, `table.py`.
  - verify: PNG saved; markdown table prints; can be regenerated from CSVs alone.
- [ ] T7 (R*): full sweep on real models.
  - verify: numeric asserts in R1..R5 pass.
- [ ] T8 (R*): external review of figure + table.
  - verify: review doc saved under docs/spec/.

## Context

Calibration target: p95 per-token KL(steered || base) = 1 nat over T tokens (T in {20, 50}), N=4 calibration prompts under greedy decoding.

Branch-pmass procedure: at fork points t in {0, 5, ..., 50} take steered prefix of length t, append `\nAnswer (true/false): ` then `{"value": ` then teacher-force one forward under steered model, sum probabilities of token variants for `true` and `false`.

Target-axis: a single contrastive pair-set built into the repo (sentiment positive vs negative or refusal yes vs no), 4 prompts each. Target Delta logit = mean over held-out prompts of difference in logit on the target token.

## Log

(append-only; only entries that change a future task)

## TODO

(out-of-scope ideas; not commitments)

## Errors

| Task | Error | Resolution |
|------|-------|------------|
