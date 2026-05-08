# iso-kl-figure

How strongly should we steer a model? How do we compare steering methods when one might be strong and one weak? These are calibration questions.

Treat steering as an intervention: we want maximum behavior change with minimum side effects, like incoherence, format collapse, or random off-target damage.

A useful analogy is a car on the road. A small nudge to the steering wheel gets corrected by the driver. A larger nudge causes a lane change. A very large nudge causes a crash the driver cannot recover from. Some crashes happen immediately. Some take a few seconds to develop, after the driver tries and fails to correct.

We measure the distribution shift caused by steering, especially the worst 5% of per-token KL that would cause a "crash", and find the largest scalar coefficient C that keeps that 5% below a safe threshold (default: 1 nat). At `alpha = 1` the steering is delivering exactly that calibrated dose. At `alpha = 2` it is doing twice the dose. *Iso-KL* means: comparing two methods at the same alpha = 1 means they are spending the same per-token KL budget, so any behavioural difference is not just one of them being louder.

Then comes the survival question. If the calibration only looked at the first w tokens of the rollout, do crashes still happen later? Most trajectories either stabilize or go off track within the first 50-100 tokens, but we want to see the long tail too. So at every fork t we ask: can the model still produce one of two valid answer tokens (`true` or `false`) when forced into a JSON-schema prefill? If yes, *alive* at t. If the probability mass on those tokens drops below 0.95, *dead*, and dead stays dead. Survival S(t) is the fraction still alive at token t, with right-censoring for rollouts that hit EOS before t.

Headline result on Qwen3.5-0.8B (`mean_diff`, w=512, n=8 held-out prompts):

- `alpha` in {0, 0.25, 0.5, 0.75, 1.0, 1.5}: 0 of 8 trajectories die at any fork.
- `alpha = 2.0`: 8 of 8 die. Median death time t = 64.
- `alpha = 4.0`: 8 of 8 die at t = 0.

A phase transition at roughly 2x the iso-KL coefficient. The calibrated dose has long-horizon headroom; doubling the dose is a slow crash; quadrupling it is an instant one. Figures and table below.


## Spaghetti: per-token KL trajectories, coloured by survival

![KL trajectories coloured by survival, Qwen3.5-0.8B](figs/post/spaghetti_qwen35_w512.png)

*Per-token KL(steered || base) for n=8 held-out long-form prompts on Qwen3.5-0.8B (mean_diff, w=512). One panel per alpha. Each thin line is one trajectory; green = alive at that token (pmass_eval >= 0.95), red = dead. Black line is the median. Dotted horizontal: KL = 1 nat (the calibration target).*

How to read this plot, in order:

1. *alpha = 0.* KL is exactly zero everywhere. This is the base model; nothing has been added. A sanity check that the splicing pipeline does not itself perturb the logits.
2. *alpha = 0.25 to 1.0.* KL stays well below the 1-nat line for most tokens. The calibration window only required p95 = 1 nat, so most tokens are far below. All trajectories are green: the model is still producing schema-valid answers throughout. The "warm but not hot" zone.
3. *alpha = 1.5.* KL still mostly below 1 nat but with a fatter upper envelope. All trajectories still alive end-to-end on the held-out set, even though calibration only guaranteed survival up to roughly `alpha = 1`. Suggests headroom.
4. *alpha = 2.0.* KL median sits around 1 nat for most of the rollout, with a noisy plateau. The trajectories turn red part-way through. Steering is now strong enough to break format. The phase transition.
5. *alpha = 4.0.* Every trajectory is red from t = 0 and KL is roughly 4-6 nats throughout. The model is no longer answering the question; it is generating something else.

The diagnostic shape is the alpha = 2 panel: KL is only modestly above the calibration target (less than 2x in nats) but format collapse is total. KL is necessary but not sufficient as a steering budget. Going from "calibrated" to "calibrated x 2" is not a smooth degradation; it crosses a cliff.

## Survival: when do trajectories die?

![Kaplan-Meier survival on pmass_eval, Qwen3.5-0.8B](figs/post/survival_qwen35_w512.png)

*Kaplan-Meier S(t) on the pmass_eval death event (pmass_eval < 0.95). One curve per alpha, n = 8 trajectories each. Dotted vertical: t = 20, the upper end of the calibration window's relevance for the answer span; everything to the right is generalization.*

| alpha | n | died | censored | S(mid) | S(end) | t at S<=0.5 |
|------:|--:|-----:|---------:|-------:|-------:|------------:|
|  0.00 | 8 |    0 |        0 |  1.000 |  1.000 |          -- |
|  0.25 | 8 |    0 |        0 |  1.000 |  1.000 |          -- |
|  0.50 | 8 |    0 |        0 |  1.000 |  1.000 |          -- |
|  0.75 | 8 |    0 |        0 |  1.000 |  1.000 |          -- |
|  1.00 | 8 |    0 |        0 |  1.000 |  1.000 |          -- |
|  1.50 | 8 |    0 |        0 |  1.000 |  1.000 |          -- |
|  2.00 | 8 |    8 |        0 |  0.586 |  0.321 |          64 |
|  4.00 | 8 |    8 |        0 |  0.000 |  0.000 |           0 |

How to read this table, in order:

1. *Censored is always 0* here, so survival numbers are not biased by short rollouts. Every trajectory reaches every fork t in `{0, 5, ..., w}`.
2. *alpha 0 to 1.5*: `died = 0`, `S = 1.0` everywhere. Format holds for the full rollout. The held-out set generalizes past the calibration window (t = 20 dotted line) all the way to t = 512. The 1-nat KL budget set on calibration prompts continues to be a survivable budget on a different prompt set, more than an order of magnitude past where calibration looked.
3. *alpha = 2.0*: `died = 8` of 8 within the rollout; median death at t = 64. All trajectories eventually break, but they survive the first 64 tokens, about 3x the calibration window. Format collapse is gradual at this dose.
4. *alpha = 4.0*: median death at t = 0. The first fork already shows `pmass_eval < 0.95`. Format collapse is immediate.
5. *Read t at S<=0.5 as a half-life.* It scales nonlinearly with alpha: 64 tokens at 2x, 0 tokens at 4x. Doubling the dose past the calibration point does not double the half-life; it eliminates it.

The calibrated alpha (`alpha = 1`) earns a high-confidence survival claim within this experiment: 8 of 8 trajectories survived all 512 forks on a held-out prompt set. Doubling the dose still leaves a usable window. Quadrupling it does not.

## Reproduce

```bash
uv sync --extra all
just smoke         # tiny-random model, ~1 min CPU
just calibrate     # one (model, method, seed, window) cell
just trajectory
just plot
```

The headline run for the figures here:

```bash
uv run --extra all python scripts/run_cell.py \
  --model Qwen/Qwen3.5-0.8B --method mean_diff --seed 0 --window 512 \
  --out-root outputs/qwen35_w512_dense \
  --run-id Qwen3.5-0.8B_mean_diff_s0_w512_dense \
  --compute-pmass --skip-pmass-qa --fork-log \
  --alphas 0.0 0.25 0.5 0.75 1.0 1.5 2.0 4.0 \
  --render-figs --render-threshold 0.95
```

Outputs land under `outputs/qwen35_w512_dense/<run-id>/figs_auto/{survival,spaghetti,aggregate}/`.

## What this repo is and isn't

In scope:
- One figure family (spaghetti, survival, aggregate) and one calibration script.
- 3 methods: `mean_diff`, `directional_ablation`, `pca`.
- A `branch_pmass` metric: fork-and-teacher-force probability mass on a forced-choice answer token after schema prefill.

Out of scope:
- A method zoo beyond those three.
- A norm-matching baseline. Iso-KL says nothing about whether two methods of equal KL move the same distance in activation space.
- A threshold sweep. The pmass < 0.95 cutoff is a single point; neighbours might disagree.

## Per-trajectory KL normalization (OLMo-2 1B, w=4096)

The figures above use raw per-token KL in nats. That works when every panel shares an alpha-scale, but cross-alpha comparison gets harder once trajectories vary in their own baseline KL noise (some prompts are just spikier than others, even at alpha = 1).

So we re-plot KL divided by each trajectory's own p95 over the first 20 tokens. The dashed reference at y = 1 is then the calibration scale by construction. Values above 1 are excursions above that trajectory's own early-rollout budget. The y-axis is shared across panels so the alpha sweep reads as one picture.

![KL/p95 trajectories, OLMo-2 1B, w=4096, n=24](figs/spaghetti_olmo2_1b_w4096_norm.png)

*Per-token `KL(steered || base) / p95(KL[:20])` for OLMo-2-0425-1B (mean_diff, w=4096, n=24 held-out prompts). One panel per alpha. Blue = alive at that token (`pmass_eval >= 0.75` at the nearest fork s <= t); orange = dead (irreversible). Black tick = EOS (right-censored). Dashed = calibration scale = 1. Black line = median across alive+dead trajectories. Threshold here is 0.75, looser than the 0.95 used for Qwen, because OLMo's early pmass is noisier.*

How to read it, in order:

1. *alpha = 0.* No steering, KL = 0, denom = 0, all trajectories skipped. Empty panel is correct.
2. *alpha = 0.25 to 0.75.* Median sits well below 1. Most trajectories never approach the calibration scale even past 4000 tokens. The 1-nat budget set on a w = 4096 calibration window has long-horizon headroom for OLMo-2 1B.
3. *alpha = 1.0.* Median hovers a bit under 1 with one outlier traveller above 2. Iso-KL is doing what it claims at the trajectory-median level on a different prompt set than calibration saw.
4. *alpha = 1.5 to 2.0.* Median crosses 1 and stays there. Most trajectories die. KL is now sustainably over budget.
5. *alpha = 4.0.* Every trajectory dies. Median rides above 1 throughout. Same instant-collapse story as the Qwen panel.

What surprised me, and what doesn't yet have an explanation:

- *Random dead traces at low alpha.* At alpha = 0.25 to 0.75, KL stays comfortably under 1 for almost every token, and yet 7 to 12 of 24 trajectories register a death event at some fork. They are not dying because the budget was exceeded; they are dying for some other reason that p95 KL does not see. Possibilities: a single high-KL spike at one token that p95 averages away; a low-KL but format-fragile region where small logit shifts knock pmass below 0.75; calibration noise on the pmass side (threshold 0.75 is loose, and OLMo's base pmass at some forks is already close to it). I don't know which yet.
- *alpha = 1.0 has fewer deaths than alpha = 0.75 (11 vs 12).* Within noise at n = 24, but the monotonicity of "more steering = more death" is not crisp at this scale.
- *Censoring drops to ~0 above alpha = 0.75.* Steering past the calibrated dose seems to suppress EOS, so trajectories run to the window length even when off-format. That makes died counts more comparable across high-alpha panels but means low-alpha "censored" is partly the model's own brevity, not the steering's.

So: per-trajectory normalization makes the cross-alpha comparison cleaner, and confirms iso-KL is roughly delivering its claimed dose on held-out prompts. It also makes the "KL is necessary but not sufficient" point sharper: the low-alpha panels show death events at well under the calibration scale.

Reproduce:

```bash
uv run --extra all python scripts/spaghetti_kl_alive.py \
  --runs-root outputs/olmo1b_w4096_dense/_OLMo-2-0425-1B_mean_diff_s0_w4096_dense_n24_single \
  --out outputs/olmo1b_w4096_dense/OLMo-2-0425-1B_mean_diff_s0_w4096_dense_n24/figs_thr075_quantile_shaded/spaghetti_norm \
  --window 4096 --threshold 0.75 --model-contains OLMo-2-0425-1B \
  --roll 301 --line-lw 0.4
```

Toggle off normalization with `--no-normalize-kl`; change the calibration window with `--calib-tokens N`.

## Honest caveats

- *Iso-KL is one fairness criterion, not the criterion.* Matching p95 per-token KL means matching distributional disagreement under greedy decoding in the calibration window. It does not match intervention norm, behavioural effect size, or human-perceived quality.
- *Pmass is a proxy.* It scores whether the model still produces one of two specific tokens after we splice in a schema. A model that has gone off-format but is otherwise coherent gets scored as dead.
- *n = 8.* Held-out prompts, not held-out seeds. The phase-transition shape is consistent at this scale; the exact half-life at alpha = 2 is not.
- *One model so far.* Gemma 4B, Gemma 12B (4-bit), OLMo-2 1B, and OLMo-3 7B at w = 4096 are queued. The phase-transition story may or may not survive scaling.


## Glossary

Operational definitions used in this repo, not textbook ones. Skim and come back as needed.

- *steering coefficient (alpha)*: scalar multiplier on the steering vector added to residual-stream activations at one layer. `alpha = 1.0` is the iso-KL-calibrated coefficient C. `alpha = 0` is the unsteered base model.
- *iso-KL calibration*: bisection on alpha such that the 95th-percentile per-token `KL(steered || base)` over a w-token calibration window equals 1 nat. Output is `c_star`, the coefficient at `alpha = 1`.
- *p95 KL*: 95th percentile of per-token `KL(steered || base)` over the calibration window. The "worst 5%" measure used by calibration.
- *window (w)*: length in tokens of the calibration rollout. `w = 512` for the figures here.
- *pmass*: probability the model puts on the set `{true, false}` after we splice in a JSON schema prefill (`'\nI should answer now.</think>{"value": '`) at fork token t.
- *pmass_eval*: pmass on a held-out prompt set, at every fork t.
- *fork point t*: the token in the rollout where we splice the schema and read pmass.
- *death*: irreversible. Alive at t means `pmass_eval(t) >= 0.95`. First time it drops below, dead, and stays dead. Below 0.95 the JSON schema stops being a stable attractor.
- *right-censoring*: the rollout hit EOS at length L < t, so we never see fork t. Drops out of the at-risk denominator from L onward; not death.
- *survival S(t)*: Kaplan-Meier estimator of the fraction alive at fork t.
- *mean_diff*: steering vector built as the difference of mean activations on a contrastive prompt pair.
- *spaghetti plot*: every individual KL trajectory drawn as one thin line, alive segments green, dead red, black line is the median.