# AGENTS.md

Inherits conventions from sibling project `steering-lite`. Read [../steering-lite/AGENTS.md](../steering-lite/AGENTS.md) if it exists.

## House rules

- Fail fast. No defensive programming, no fallbacks, no silent dequant.
- Keep this repo small. Anything beyond the headline figure + table belongs in another repo.
- Use `einops` and `jaxtyping` shape annotations at function boundaries only. Tensor dim letters: `b s d` (batch, seq, d_model), `n` (prompts), `t` (token positions), `f` (fork points).
- No backward compat.
- Single functional smoke test = the real pipeline at tiny scale (`tests/test_smoke.py`).
- Methods register via `@register_config` and `@register` decorators; mirror `steering-lite/src/steering_lite/config.py`.
- All experiment scripts write CSV/TSV. Plot/table scripts read CSV/TSV. Never plot from in-memory state.

## Out of scope (deliberately)

- Method zoo beyond mean_diff, directional_ablation, pca.
- LessWrong post / paper draft.
- Citation collection.
- tinymfv or any external eval dependency.

## Verify

`just smoke` -> 3/3 methods pass calibrate -> trajectory -> branch-pmass on tiny-random Llama. Asserts nonzero KL at coeff>0, zero KL at coeff=0, branch-pmass in [0,1].
