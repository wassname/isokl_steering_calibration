# iso-kl-figure

Minimal repo with one job: produce a figure and a table that demonstrate iso-KL calibration is stable across models, seeds, and calibration windows.

## Claim (narrow)

Calibrating a steering coefficient so that p95 per-token KL(steered || base) hits 1 nat in a short calibration window:

- C1: bisection converges for every method tested; held-out p95 KL lands near 1 nat.
- C2 (not too cold): target-axis Delta logit at calibrated alpha is non-zero across methods.
- C3 (not too hot): base-NLL of generated text and branch-pmass of a forced format token stay near base at calibrated alpha across methods.

The 2x check is a sanity probe, not a margin claim. Reported as: at 2x, p95 KL exceeds 1 nat for N of M cells.

Honesty footnote: matched on per-token distributional disagreement under greedy decoding in the calibration window. This is one defensible notion of fairness; not equivalence on intervention norm or behavioral effect size.

## Quick start

```bash
uv sync --extra all
just smoke         # tiny-random model, ~1 min CPU
just calibrate     # one (model, method, seed, window) cell
just trajectory
just table
just plot
just table-md
```

`just sweep` runs the full grid (3 models x 3 methods x 3 seeds x 2 windows) used by Figure 1.

## What this repo does NOT do

- No paper or LessWrong draft.
- No method zoo beyond mean_diff, directional_ablation, pca.
- No threshold sweep, no calibration-set-size sweep.
- No tinymfv integration. Target-axis is a single contrastive sentiment / refusal pair.
